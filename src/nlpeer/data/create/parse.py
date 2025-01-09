import copy
import logging
from copy import deepcopy
from typing import Any, Optional, Tuple, List, Dict
from xml.etree import ElementTree

import os.path

from grobid_client.grobid_client import GrobidClient, ServerUnavailableException
from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype, SpanNode
from intertext_graph.itsentsplitter import SentenceSplitter, IntertextSentenceSplitter, make_sentence_nodes
from intertext_graph.parsers.f1000_xml_parser import F1000XMLParser
from intertext_graph.parsers.itparser import IntertextParser

#
# Constants
#


# GROBID
from lxml import etree

from nlpeer import NTYPES, NTYPE_TITLE, NTYPE_HEADING, NTYPE_PARAGRAPH, NTYPE_ABSTRACT, NTYPE_LIST, NTYPE_LIST_ITEM, \
    NYTPE_ELEMENT_REFERENCE, NTYPE_BIB_REFERENCE, NTYPE_HEADNOTE, NTYPE_FOOTNOTE, NTYPE_FIGURE, NTYPE_TABLE, \
    NTYPE_FORMULA, NTYPE_MEDIA, NTYPE_BIB_ITEM

GROBID_CONF = {}
GROBID_HOST = os.environ.get("GROBID_HOST")
if GROBID_HOST:
    GROBID_CONF["grobid_server"] = GROBID_HOST
GROBID_PORT = os.environ.get("GROBID_PORT")
if GROBID_PORT:
    GROBID_CONF["grobid_port"] = GROBID_PORT

# todo: currently the bibliography is not parsed as part of the text, but as isolated nodes. Might make sense to add!

#
# Classes
#


class TEIXMLParser(IntertextParser):
    """
    Parser to transform a TEI XML document into an IntertextDocument.

    Author: Nils Dycke
    Co-author: Jan-Micha Bodensohn (scaffolding and base paragraph parser)
    """

    def __init__(self, xml_file_path: str):
        """
        Initialize the TEIXMLParser for a particular paper.

        :param xml_file_path: filepath of the TEI XML file (GROBID output)
        """
        super(TEIXMLParser, self).__init__(xml_file_path)
        self._xml_file_path: str = xml_file_path

    def __call__(self) -> IntertextDocument:
        """
        Parse the TEI XML document into an IntertextDocument.

        :return: the IntertextDocument
        """
        return self._parse_document()

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        raise NotImplementedError  # TODO: implement this

    @staticmethod
    def _parse_section_content(section, prefix):
        sub_graph = IntertextDocument([], [], prefix)
        graph_refs = []
        figures = {}

        # add artificial root node
        root = Node("root", ntype=NTYPE_TITLE)
        sub_graph.add_node(root)

        predecessor = root
        list_parent = None
        for child in section:
            # add paragraphs
            if child.tag == f"{prefix}p":
                paragraph = child

                # get text
                paragraph_text, children_ix = TEIXMLParser._flatten_xml_element_with_child_ix(paragraph)
                TEIXMLParser._check_text(paragraph_text, "paragraph")

                # get references in text
                reference_ixs = [(t, pre_ix, pre_ix + contl) for (t, pre_ix, contl) in children_ix if
                                 t.tag == f"{prefix}ref"]

                # ACL layout specific --> lists start with bulletpoints
                if paragraph_text[0] == "\u2022":
                    if list_parent is None:
                        list_parent = TEIXMLParser._add_node(sub_graph, "", NTYPE_LIST, predecessor=predecessor,
                                                             parent=root)
                        predecessor = list_parent
                    li = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_LIST_ITEM, predecessor=predecessor,
                                                parent=list_parent)
                    predecessor = li
                else:
                    if list_parent is not None:
                        # no list type paragraph any more: erase existing list parent
                        list_parent = None

                    p = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_PARAGRAPH, predecessor=predecessor,
                                               parent=root)
                    predecessor = p

                if len(reference_ixs) > 0:
                    graph_refs += [(predecessor, reference_ixs)]

            elif child.tag == f"{prefix}figure":
                fxid, figure = TEIXMLParser._parse_figure(sub_graph, child, prefix, predecessor, root)
                predecessor = figure
                figures[fxid] = figure
            elif child.tag == f"{prefix}head":
                # head -- ignoring them
                pass
            elif child.tag == f"{prefix}formula":
                # get text
                formula_text = TEIXMLParser._flatten_xml_element(child)
                TEIXMLParser._check_text(formula_text, "formula")

                p = TEIXMLParser._add_node(sub_graph, formula_text, NTYPE_FORMULA, predecessor=predecessor, parent=root)
                predecessor = p
            else:
                # add the rest
                logging.info(f"UNKNOWN TAG {section.tag} within section -- adding as flattend version.")
                paragraph_text, children_ix = TEIXMLParser._flatten_xml_element_with_child_ix(child)
                TEIXMLParser._check_text(paragraph_text, "paragraph")

                # get references in text
                reference_ixs = [(t, pre_ix, pre_ix + contl) for (t, pre_ix, contl) in children_ix if
                                 t.tag == f"{prefix}ref"]

                p = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_PARAGRAPH, predecessor=predecessor,
                                           parent=root)
                predecessor = p

                if len(reference_ixs) > 0:
                    graph_refs += [(predecessor, reference_ixs)]

        return sub_graph, predecessor, graph_refs, figures

    def _parse_abstract(self, doc, abstract, prefix, predecessor, article_title):
        abstract_title = self._add_node(doc, "Abstract", NTYPE_ABSTRACT, predecessor=predecessor, parent=article_title)
        predecessor = abstract_title

        if abstract is None:
            return abstract_title, predecessor

        content = ""
        merge = False
        for child in abstract:
            # get text (including any possible children)
            text = self._flatten_xml_element(child)
            self._check_text(text, "abstract paragraph")

            content += text

            if child.tag != f"{prefix}p":
                # if non-paragraph, treat as hick-up by GROBID and append to content of next node and force merge
                logging.info(f"Unexpected paragraph tag '{child.tag}' in the abstract! Treating as in-line text.")

                merge = True
                content += " "
            elif merge and predecessor.ntype == NTYPE_PARAGRAPH:
                # if a previous element was erroneously inserted, merge this paragraph with the previous one
                predecessor.content += " " + text

                merge = False
                content = ""
            else:
                # no merging required or cannot merge with previous node: simply add a new one with all content
                predecessor = self._add_node(doc, content, NTYPE_PARAGRAPH, predecessor=predecessor,
                                             parent=abstract_title)

                merge = False
                content = ""

        # if there is still content left (no paragraph follows a hick-up)
        if len(content) > 0:
            predecessor = self._add_node(doc, content.strip(), NTYPE_PARAGRAPH, predecessor=predecessor,
                                         parent=abstract_title)

        return abstract_title, predecessor

    @staticmethod
    def _parse_figure(doc, figure, prefix, predecessor, parent):
        head = figure.find(f"{prefix}head")
        label = figure.find(f"{prefix}label")
        xid = figure.get('{http://www.w3.org/XML/1998/namespace}id')
        figDesc = figure.find(f"{prefix}figDesc")

        if head is None:
            return None, None

        if "type" in figure.attrib and figure.attrib["type"] == "table":
            table = figure.find(f"{prefix}table")

            table_node = TEIXMLParser._add_node(doc,
                                                head.text if head.text is not None else "",
                                                NTYPE_TABLE,
                                                meta={"label": label.text if label is not None else None,
                                                      "id": xid,
                                                      "caption": figDesc.text if figDesc is not None else None},
                                                predecessor=predecessor,
                                                parent=parent)
            predecessor = table_node

            table_content = TEIXMLParser._add_node(doc,
                                                   str(ElementTree.tostring(table)),
                                                   NTYPE_MEDIA,
                                                   meta={},
                                                   predecessor=predecessor,
                                                   parent=table_node)
            predecessor = table_content
        else:
            graphic = figure.find(f"{prefix}graphic")

            figure_node = TEIXMLParser._add_node(doc,
                                                 head.text if head.text is not None else "",
                                                 NTYPE_FIGURE,
                                                 meta={"label": label.text if label is not None else None,
                                                       "id": xid,
                                                       "caption": figDesc.text if figDesc is not None else None},
                                                 predecessor=predecessor,
                                                 parent=parent)
            predecessor = figure_node

            figure_content = TEIXMLParser._add_node(doc,
                                                    TEIXMLParser._flatten_xml_element(graphic) if graphic else "",
                                                    NTYPE_MEDIA,
                                                    meta={},
                                                    predecessor=predecessor,
                                                    parent=figure_node)
            predecessor = figure_content

        return xid, predecessor

    @staticmethod
    def _parse_bibitem(bib_item, prefix):
        xid = bib_item.get('{http://www.w3.org/XML/1998/namespace}id')
        publishing_info = bib_item.find(f"{prefix}monogr")
        paper_info = bib_item.find(f"{prefix}analytic")

        # parse publishing information
        if publishing_info:
            pub_title = publishing_info.find(f"{prefix}title")
            pub_title = pub_title.text if pub_title is not None else None

            pub = publishing_info.find(f"{prefix}imprint/{prefix}publisher")
            pub = pub.text if pub is not None else None

            pub_date = publishing_info.find(f"{prefix}imprint/{prefix}date")
            pub_date = pub_date.attrib["when"] if pub_date is not None and "when" in pub_date.attrib else None
        else:
            pub_title = None
            pub = None
            pub_date = None

        # add paper information if present
        if paper_info:
            title = paper_info.find(f"{prefix}title")
            title = title.text if title is not None else None

            authors = paper_info.findall(f"{prefix}author/{prefix}persName")

            author_names = []
            for a in authors:
                forename = a.find(f'{prefix}forename')
                surname = a.find(f'{prefix}surname')
                canonical_author = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}"

                author_names += [canonical_author] if len(canonical_author) > 0 else []
        else:
            title = None
            author_names = None

        return xid, title, author_names, pub_title, pub, pub_date

    @staticmethod
    def _flatten_xml_element(element):
        stack = [(element, -1, None)]

        while True:
            elem, pred, content = stack.pop(-1)

            # visiting node the first time
            if content is None:
                content = elem.text if elem.text else ""

                # revisit element after children
                stack += [(elem, pred, content)]

                # add children
                stack += reversed([(child, len(stack) - 1, None) for child in elem])
            else:
                # revisiting the node (content is set)
                suffix = elem.tail if elem.tail else ""

                if pred >= 0:
                    pre_elem, pre_pred, pre_content = stack[pred]
                    stack[pred] = (pre_elem, pre_pred, pre_content + content + suffix)
                else:
                    # terminating at parent most element
                    return content + suffix

                # don't add to stack again
        # should always terminate

    @staticmethod
    def _flatten_xml_element_with_child_ix(element):
        # xml element, predecessor ix, pred. merged ixs, parsed content
        stack = [(element, -1, [], None)]

        while True:
            elem, pred, mergedix, content = stack.pop(-1)

            # visiting node the first time
            if content is None:
                content = elem.text if elem.text else ""

                # revisit element after children
                stack += [(elem, pred, mergedix, content)]

                # add children
                stack += reversed([(child, len(stack) - 1, [], None) for child in elem])
            else:
                # revisiting the node (content is set)
                suffix = elem.tail if elem.tail else ""

                if pred >= 0:
                    pre_elem, pre_pred, pre_mergedix, pre_content = stack[pred]

                    new_mergedix = pre_mergedix + \
                                   [(elem, len(pre_content), len(content))] + \
                                   [(t, prel + len(pre_content), contl) for (t, prel, contl) in mergedix]

                    stack[pred] = (pre_elem, pre_pred, new_mergedix, pre_content + content + suffix)
                else:
                    # terminating at parent most element
                    return content + suffix, mergedix

                # don't add to stack again
        # should always terminate

    @staticmethod
    def _add_node(doc, content, ntype, meta=None, predecessor=None, parent=None):
        new_node = Node(
            content=content,
            ntype=ntype,
            meta=meta
        )
        doc.add_node(new_node)

        if parent is not None:
            parent_edge = Edge(
                src_node=parent,
                tgt_node=new_node,
                etype=Etype.PARENT
            )
            doc.add_edge(parent_edge)

        if predecessor is not None:
            next_edge = Edge(
                src_node=predecessor,
                tgt_node=new_node,
                etype=Etype.NEXT
            )
            doc.add_edge(next_edge)

        return new_node

    @staticmethod
    def _add_subtree(doc, subTree, lastSubTree, targetParent, targetPredecessor):
        # add nodes
        new_nodes = {}
        for n in subTree.nodes:
            new_n = TEIXMLParser._add_node(doc, n.content, n.ntype, n.meta)
            new_nodes[n.ix] = new_n

        # add edges
        for e in subTree.edges:
            new_e = Edge(
                src_node=new_nodes[e.src_node.ix],
                tgt_node=new_nodes[e.tgt_node.ix],
                etype=e.etype
            )
            doc.add_edge(new_e)

        # get pseudo root and replace parent edges
        new_pseudo_root = new_nodes[subTree.root.ix]
        for ce in new_pseudo_root.get_edges(Etype.PARENT, outgoing=True, incoming=False):
            new_parent = Edge(
                src_node=targetParent,
                tgt_node=ce.tgt_node,
                etype=Etype.PARENT
            )
            doc.add_edge(new_parent)
            doc.remove_edge(ce)

        # if no other nodes except root: skip the next part
        new_pseudo_next_edges = new_pseudo_root.get_edges(Etype.NEXT, outgoing=True, incoming=False)
        if len(new_pseudo_next_edges) > 0:
            new_pseudo_next = new_pseudo_next_edges[0]
            new_next = Edge(
                src_node=targetPredecessor,
                tgt_node=new_pseudo_next.tgt_node,
                etype=Etype.NEXT
            )
            doc.add_edge(new_next)
            doc.remove_edge(new_pseudo_next)

        doc.remove_node(new_pseudo_root)

        # output
        start_subtree = targetParent
        end_subtree = new_nodes[lastSubTree.ix] if len(new_pseudo_next_edges) > 0 else None

        return start_subtree, end_subtree, new_nodes

    def _parse_document(self) -> IntertextDocument:
        """
        Parse the given TEI XML document.

        :return: resulting IntertextDocument
        """
        # create intertext document
        prefix = self._xml_file_path.split(os.path.sep)[-1]

        itg_doc = IntertextDocument(
            nodes=[],
            edges=[],
            prefix=prefix
        )

        # the content of the document is completely derived from the TEI XML file
        tree = ElementTree.parse(self._xml_file_path)
        prefix = "{http://www.tei-c.org/ns/1.0}"

        # create article title as root
        title = tree.getroot().find(f"{prefix}teiHeader/{prefix}fileDesc/{prefix}titleStmt/{prefix}title").text
        title = title if title is not None else ""
        article_title_node = self._add_node(itg_doc, title, NTYPE_TITLE)
        predecessor = article_title_node

        #
        # PARSE THE ABSTRACT
        #
        abstract = tree.getroot().find(f"{prefix}teiHeader/{prefix}profileDesc/{prefix}abstract/{prefix}div")
        abstract_title_node, predecessor = self._parse_abstract(itg_doc, abstract, prefix, predecessor,
                                                                article_title_node)

        #
        # PARSE BODY
        #
        body = tree.getroot().find(f"{prefix}text/{prefix}body")

        body_refs = []
        body_figs = {}
        content_graph = []
        last_section_title = []
        for section in body.findall(f"{prefix}div"):
            content, last_elem, refs, figures = self._parse_section_content(section, prefix)

            head = section.find(f"{prefix}head")
            if head is None:
                logging.info(f"Div without a heading in {self._xml_file_path}.")

            # empty section
            if len(content.edges) == 0 and len(content.nodes) <= 1 and head is None:
                logging.info(f"Encountered empty section in {self._xml_file_path}.")

            content_graph += [(content, last_elem, refs, figures)]

            # fixme currently erroneous head nodes (e.g. with text, but without number) are discarded entirely
            if head is not None and "n" in head.attrib:
                # pop current content
                current_content, current_last, current_refs, current_figures = content_graph.pop(-1)

                # add previous contents to the predecessor if existent, else create a dummy section first
                if len(content_graph) > 0:
                    if len(last_section_title) == 0:
                        dummy_node = self._add_node(itg_doc, "", NTYPE_HEADING, {"section": "1"},
                                                    predecessor=article_title_node, parent=article_title_node)
                        last_section_title += [dummy_node]
                        predecessor = dummy_node

                    pred_parent = last_section_title[-1]
                    for c, l, r, f in content_graph:
                        sub_root, sub_last, node_map = self._add_subtree(itg_doc, c, l, pred_parent, predecessor)
                        predecessor = sub_last if sub_last is not None else predecessor

                        mapped_refs = [(node_map[n.ix], ref) for n, ref in r]
                        body_refs += mapped_refs

                        mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in f.items()}
                        body_figs.update(mapped_figs)

                    # reset content stack -- added all previous contents
                    content_graph = []

                # get section name and number
                section_name = head.text
                section_n = head.attrib["n"]
                self._check_text(section_name, "section title")

                # find parent node
                section_parent_node = None
                if len(last_section_title) == 0:
                    # is first section
                    section_parent_node = article_title_node
                else:
                    for st in last_section_title:
                        st_n = st.meta["section"]

                        if self._is_child_section_count(section_n, st_n):
                            section_parent_node = st
                            break

                    if section_parent_node is None:
                        section_parent_node = article_title_node

                # add new section title with content
                section_title_node = self._add_node(itg_doc,
                                                    section_name,
                                                    NTYPE_HEADING,
                                                    {"section": section_n},
                                                    predecessor=predecessor,
                                                    parent=section_parent_node)
                predecessor = section_title_node
                last_section_title += [section_title_node]

                sub_root, sub_last, node_map = self._add_subtree(itg_doc, current_content, current_last,
                                                                 section_title_node,
                                                                 predecessor)
                predecessor = sub_last if sub_last is not None else predecessor

                mapped_refs = [(node_map[n.ix], r) for n, r in refs]
                body_refs += mapped_refs

                mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in current_figures.items()}
                body_figs.update(mapped_figs)

        # add left-over content-graph elements
        if len(content_graph) > 0:
            if len(last_section_title) == 0:
                dummy_node = self._add_node(itg_doc, "", NTYPE_HEADING, {"section": "1"},
                                            predecessor=article_title_node, parent=article_title_node)
                last_section_title += [dummy_node]
                predecessor = dummy_node

            pred_parent = last_section_title[-1]
            for c, l, r, f in content_graph:
                sub_root, sub_last, node_map = self._add_subtree(itg_doc, c, l, pred_parent, predecessor)
                predecessor = sub_last if sub_last is not None else predecessor

                mapped_refs = [(node_map[n.ix], ref) for n, ref in r]
                body_refs += mapped_refs

                mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in f.items()}
                body_figs.update(mapped_figs)

        for figure in body.findall(f"{prefix}figure"):
            fxid, figure_node = self._parse_figure(itg_doc, figure, prefix, predecessor, article_title_node)
            predecessor = figure_node
            body_figs[fxid] = figure_node

        #
        ## PARSE BACK MATTER
        #
        back = tree.getroot().find(f"{prefix}text/{prefix}back")

        bibliography = {}
        for bib_item in back.findall(f"{prefix}div/{prefix}listBibl/{prefix}biblStruct"):
            xid, title, authors, pub_title, pub, pub_date = self._parse_bibitem(bib_item, prefix)

            bib_node = self._add_node(itg_doc, f"{', '.join(authors) if authors is not None else 'UNKNOWN'}, "
                                               f"{title}, "
                                               f"{pub_date if pub_date else ''}, "
                                               f"{pub_title if pub_title else ''}, "
                                               f"{pub if pub else ''}.",
                                      ntype=NTYPE_BIB_ITEM,
                                      meta={"xid": xid,
                                            "authors": authors,
                                            "title": title,
                                            "pub_date": pub_date,
                                            "pub_title": pub_title,
                                            "pub": pub})
            bibliography[xid] = bib_node

        #
        ## ADD REFERENCES
        #
        for node, refs in body_refs:
            for r in refs:
                xml_elem, start, end = r
                rtype = xml_elem.attrib["type"]

                # skip invalid references with missing target (for now)
                if "target" not in xml_elem.attrib:
                    continue

                rtarget = xml_elem.attrib["target"][1:]

                # add span node
                ref_node = SpanNode(
                    ntype=NTYPE_BIB_REFERENCE if rtype == "bibr" else NYTPE_ELEMENT_REFERENCE,
                    src_node=node,
                    start=start,
                    end=end,
                    meta={"from_xml_type": rtype, "from_xml_target": rtarget}
                )
                itg_doc.add_node(ref_node)

                # add link (where possible)
                target_node = None
                if rtype == "bibr" and rtarget in bibliography:
                    target_node = bibliography[rtarget]
                elif (rtype == "figure" or rtype == "table") and target_node in body_figs:
                    target_node = body_figs[rtarget]

                if target_node:
                    link = Edge(ref_node, target_node, etype=Etype.LINK)
                    itg_doc.add_edge(link)

        return itg_doc

    @staticmethod
    def _check_text(text: str, element_name: str) -> str:
        assert isinstance(text, str), f"{element_name} is not a string, but a {type(text)}!"
        return text

    @staticmethod
    def _compare_section_counts(cntA: str, cntB: str) -> int:
        nAs = cntA.split(".")
        nBs = cntB.split(".")

        for i, nA in enumerate(nAs):
            if len(nBs) <= i:
                return 1  # b higher level than a (a > b)
            nB = nBs[i]
            if int(nA) != int(nB):
                return -1 if int(nA) < int(nB) else 1  # a earlier than b (a < b)

        return 0 if len(nAs) == len(nBs) else -1  # a higher level than b (a < b), else equal

    @staticmethod
    def _is_child_section_count(cntA: str, cntB: str) -> bool:
        return cntA.startswith(cntB)


class F1000XMLParserBM(F1000XMLParser):
    """
    The F1000XMLParserBM is an extension of the F1000XMLParser provided by the ITG library. To standardize
    the benchmark structure, we need to adapt the given parsing strategy.

    We can translate some nodes post-hoc (assuming that the rough parsing structure is the same and semantics fit):
        article-title -> title
        title -> heading
        p -> paragraph
        abstract = abstract

        supplementary_material -> paragraph
        preformat -> paragraph
        disp-quote -> paragraph

        label && label.content.startswith("Figure") -> figure
            > fig                                   -> media

        label && label.content.startswith("Table") -> table
            > table-wrap                           -> media

        label && label.content.startswith("Algorithm") -> figure
            > boxed text

        math -> formula

        boxed-text -> figure #TODO
        table-warp -> table #todo

    We need to add or refactor other node types:
        * "ref" is bib_item, but we parse more meta-information (i.e. authors etc.)
        * lists and list-items are so-far not mapped.
        * in-line references to figures and bibitems are not tagged as span nodes so far

    Additionally, the parser delivers us meta-data on authors etc, which we should include in the benchmark. To keep
    things clean, we will decouple this.
    """

    def __init__(self, xml_file_path: str):
        """
        Initialize the F1000XMLParserBM for a particular paper.

        :param xml_file_path: filepath of the F1000 XML file
        """
        super(F1000XMLParserBM, self).__init__(xml_file_path)
        self._xml_file_path: str = xml_file_path

        self.ntype_mapping = {
            "title": NTYPE_HEADING,
            "article-title": NTYPE_TITLE,
            "p": NTYPE_PARAGRAPH,
            "supplementary-material": NTYPE_PARAGRAPH,
            "preformat": NTYPE_PARAGRAPH,
            "disp-quote": NTYPE_PARAGRAPH,
            "list": NTYPE_LIST,
            "list-item": NTYPE_LIST_ITEM,
            "disp-formula": NTYPE_FORMULA,
            "def-list": NTYPE_LIST
        }

        self.complex_ntype_mapping = [
            (F1000XMLParserBM._is_figure_ntype, F1000XMLParserBM._convert_to_figure_node),
            (F1000XMLParserBM._is_table_ntype, F1000XMLParserBM._convert_to_table_node),
            (F1000XMLParserBM._is_other_media_ntype, F1000XMLParserBM._convert_to_figure_node)
        ]

    @staticmethod
    def _is_other_media_ntype(node):
        return node.ntype == "label"

    @staticmethod
    def _is_figure_ntype(node):
        return (node.ntype == "label" and
                (node.content.lower().startswith("figure") or
                 node.content.lower().startswith("algorithm") or
                 node.content.lower().startswith("box")) or
                node.content.lower().startswith("listing")) or \
               (node.ntype == "fig") or \
               (node.ntype == "caption") or \
               (node.ntype == "boxed-text")

    @staticmethod
    def _is_table_ntype(node):
        return (node.ntype == "label" and node.content.lower().startswith("table")) or \
               ("table" in node.ntype)

    @staticmethod
    def _convert_to_figure_node(node):
        # adapt top node of type label or of type fig
        node.ntype = NTYPE_FIGURE
        node.meta = {
            "label": node.content,
            "id": node.meta["id"] if "id" in node.meta else None,
            "caption": node.meta["caption"] if "caption" in node.meta else None
        }

        # get figure child node ("the content") from top node, if existent
        fig_node = node.get_edges(Etype.PARENT, incoming=False, outgoing=True)
        if len(fig_node) > 0:
            fig_node = fig_node[0].tgt_node

            # adapt children node of type fig
            fig_node.ntype = NTYPE_MEDIA
            fig_node.meta = {
                "id": fig_node.meta["id"] if "id" in fig_node.meta else None,
                "uri": fig_node.meta["uri"] if "uri" in fig_node.meta else None
            }

    @staticmethod
    def _convert_to_table_node(node):
        # adapt top node of type label
        node.ntype = NTYPE_TABLE
        node.meta = {
            "label": node.content,
            "id": node.meta["id"] if "id" in node.meta else None,
            "caption": node.meta["caption"] if "caption" in node.meta else None
        }

        # get table child node from label node
        tbl_node = node.get_edges(Etype.PARENT, incoming=False, outgoing=True)
        if len(tbl_node) > 0:
            tbl_node = tbl_node[0].tgt_node

            # adapt children node of type table-wrap
            tbl_node.ntype = NTYPE_MEDIA
            tbl_node.meta = {
                "id": tbl_node.meta["id"] if "id" in tbl_node.meta else None,
                "uri": tbl_node.meta["uri"] if "uri" in tbl_node.meta else None
            }

    def __call__(self) -> IntertextDocument:
        """
        Parse the F1000 XML document into an IntertextDocument.

        :return: the IntertextDocument
        """
        return self._parse_document()

    def _parse_refs(self, ref_list: etree._Element) -> None:
        for ref in ref_list:
            target_xmlid = ref.attrib['id']

            label = ref.find("label")
            label = label.text if label is not None else None

            citation = ref.find("mixed-citation")

            title = citation.find("article-title")
            title = ''.join([e.strip() for e in title.itertext()]) if title is not None else None

            pub_date = citation.find("year")
            pub_date = pub_date.text if pub_date is not None else None

            if citation is not None:
                pub_info = [i for i in citation.findall("*") if i.tag not in ["article-title", "person-group", "year"]]
                pub_text = "".join([" ".join([t.strip() for t in i.itertext()]) for i in pub_info])
            else:
                pub_text = None

            authors = citation.findall("person-group[@person-group-type='author']/name")
            author_list = []
            for a in authors:
                name_components = []
                given = a.find("given-names")
                name_components += [given.text] if given is not None and given.text is not None else []
                suffix = a.find("suffix")
                name_components += [suffix.text] if suffix is not None and suffix.text is not None else []
                surname = a.find("surname")
                name_components += [surname.text] if surname is not None and surname.text is not None else []

                author_list += [" ".join(name_components)]

            bib_item = Node(f"{', '.join(author_list) if len(author_list) > 0 else 'UNKNOWN'}, " +
                            f"{title if title is not None else 'UNKNOWN'}, " +
                            f"{pub_date if pub_date is not None else ''}, " +
                            f"{pub_text if pub_text is not None else ''}",
                            ntype=NTYPE_BIB_ITEM,
                            meta={"xid": target_xmlid,
                                  "id": label,
                                  "authors": author_list,
                                  "title": title,
                                  "pub_date": pub_date,
                                  "pub": pub_text})

            self._xref_targets[target_xmlid] = bib_item

    # copied from f1000 xml parser and adapted to support empty nodes
    def _make_node(self, element: etree._Element, stringify: bool = False, meta: Dict[str, Any] = None) -> Optional[
        Node]:
        if stringify:
            content = self._stringify(element)
            if content:
                content = self._parse_whitespace(content)
            if content:
                return super(F1000XMLParser, self)._make_node(content, element.tag, meta)
        else:
            content = ""
            return super(F1000XMLParser, self)._make_node(content, element.tag, meta)

    # copied from F1000 parser and adapted
    def _split_element(cls, element: etree._Element, selector: str) -> List[etree._Element]:
        """Split an element before and after the selector."""
        node = deepcopy(element)

        children = [c for c in node]
        split_child = node.xpath(selector)[0]
        split_child_i = children.index(split_child)

        assert split_child_i != -1, "something went wrong splitting an element. Invalid selector passed."

        c_before = node[:max(split_child_i, 0)]
        c_after = node[min(split_child_i + 1, len(children)):]

        root_before = etree.Element(element.tag, element.attrib, element.nsmap)
        root_before.text = element.text
        for c in c_before:
            root_before.append(c)

        root_after = etree.Element(element.tag, element.attrib, element.nsmap)
        root_after.tail = element.tail
        for c in c_after:
            root_after.append(c)

        return [root_before, split_child, root_after]

    # copied from f1000_xml_parser.py with minor adaptions
    def _parse_element(self, element: etree._Element) -> Tuple[Optional[Node], List[etree._Element]]:
        children = list(element)
        if children:
            # Parse nodes with children, i.e. subtrees
            if element.tag == 'body':
                return None, children
            elif element.tag == 'abstract':
                node = self._make_node(element, stringify=False)
                node.content = "Abstract"
            elif element.tag == 'sec':
                # Move the title child to the root node of a section
                title_element = element.xpath('title')
                if title_element and title_element[0].text:
                    meta = {'section': self._generate_sec_index(element)}
                    if 'id' in element.attrib:
                        meta['id'] = element.attrib['id']
                    if 'sec-type' in element.attrib:
                        meta['sec-type'] = element.attrib['sec-type']
                    node = self._make_node(title_element[0], stringify=True, meta=meta)
                    children.remove(title_element[0])
                    # Keep track of potential xref targets
                    if meta and 'id' in meta:
                        self._xref_targets[meta['id']] = node
                else:
                    # Section has no title
                    node = None
            elif element.tag == 'list':
                # Concatenate list items with new line
                node = self._make_node(element, stringify=False)
                # add children
                children = element.xpath('list-item')
            elif element.tag == "list-item":
                node = self._make_node(element, stringify=True)
                children = []
            elif element.tag == "def-list":
                node = self._make_node(element, stringify=False)
                # add children
                children = element.xpath("def-item")
            elif element.tag == "def-item":
                t = element.xpath("term")
                d = element.xpath("def")

                if t and d:
                    content = f"{self._stringify(t[0])}: {self._stringify(d[0])}"
                    node = super(F1000XMLParser, self)._make_node(content, "list-item")
                    children = []
                else:
                    return None, []
            elif element.tag == 'p':
                # Stringify paragraphs, drop all inline tags
                tags = [e.tag for e in element]
                if 'boxed-text' in tags:
                    # Boxed text has to be processed before other nested types as is might contain these as children
                    return None, self._elevate_element(element, 'boxed-text')
                elif element.xpath("table-wrap"):
                    return None, self._split_element(element, "table-wrap")
                elif element.xpath('preformat'):
                    # Elevate immediate children but ignore nested inline tags
                    return None, self._elevate_element(element, 'preformat')
                elif 'list' in tags:
                    # Split paragraph before and after an inline list
                    # A human would probably read this as separate paragraphs
                    return None, self._split_element(element, 'list')
                elif "disp-formula" in tags:
                    #if len(children) > 1: #todo verify
                    #    return None, self._split_element(element, "disp-formula")
                    #else:
                    #    return self._make_node(children[0], stringify=True), []
                    return None, self._split_element(element, "disp-formula")

                meta = None
                # Add metadata for xrefs which will later be parsed into edges
                if 'xref' in tags:
                    xrefs = self._collect_xrefs(element)
                    if xrefs:
                        meta = {'xrefs': xrefs}
                # Drop inline xref
                etree.strip_tags(element, 'xref')
                node = self._make_node(element, stringify=True, meta=meta)
                # Drop children
                children = []
            elif element.tag in ['fig', 'table-wrap', 'boxed-text']:
                # Get optional meta data
                meta = self._parse_node_meta(element)
                label_element = element.xpath('label')
                caption_element = element.xpath('caption')
                if label_element:
                    # Move the label child to the root node
                    node = self._make_node(element.xpath('label')[0], stringify=True, meta=meta)
                    # Remove the label tag and do another recursive call to parse element as an XML node
                    etree.strip_elements(element, 'label', with_tail=False)
                    children = [element]
                elif caption_element:
                    # Move the caption child to the root node
                    node = self._make_node(element.xpath('caption')[0], stringify=True, meta=meta)
                    # Remove the label tag and do another recursive call to parse element as an XML node
                    etree.strip_elements(element, 'caption', with_tail=False)
                    children = [element]
                else:
                    # Handle second recursive call or cases where there is no label
                    node = self._make_xml_node(element, meta=meta)
                    # Drop children
                    children = []
                # Keep track of potential xref targets
                if meta and 'id' in meta:
                    self._xref_targets[meta['id']] = node
            elif 'formula' in element.tag:
                meta = self._parse_node_meta(element)
                node = self._make_node(element, stringify=True, meta=meta)
                children = []
            else:
                node = self._make_xml_node(element)
                # Drop children
                children = []
        else:
            # Parse leaf nodes with potential inline tags
            node = self._make_node(element, stringify=True)
        return node, children

    def _parse_document(self):
        # prep
        prefix = self._xml_file_path.split(os.path.sep)[-1]

        # parse main document with the standard parser, but exchanged sub-routines
        doc = self._parse(self._root, None, prefix)

        if doc is None:
            raise ValueError(f"The passed F1000 XML {self._xml_file_path} could not be parsed. Body not found.")

        # replace old node types by new ones, where an easy type mapping suffices
        nodes_by_ntype = [(newtype, [n for n in doc.nodes if n.ntype == oldtype]) for oldtype, newtype in
                          self.ntype_mapping.items()]
        for newtype, nodes in nodes_by_ntype:
            for n in nodes:
                n.ntype = newtype

        # realize the more complex mappings on node level
        for criterion, mapping in self.complex_ntype_mapping:
            matching_nodes = filter(criterion, doc.nodes)
            for mn in matching_nodes:
                mapping(mn)

        # verify validity
        invalid_types = set([n.ntype for n in doc.nodes if n.ntype not in NTYPES])
        if len(invalid_types) > 0:
            logging.warning(
                f"Found invalid node types in the document {self._xml_file_path} after parsing: {str(invalid_types)}")

        # discard any invalid node types
        to_remove = []
        for n in doc.nodes:
            if n in invalid_types:
                to_remove += [n]

        for n in to_remove:
            doc.remove_node(n)

        # add span nodes for citations and references
        # todo currently full paragraphs are regarded as the source of a link, we want that to be span nodes

        return doc


class F1000XMLParserMetadata(F1000XMLParser):
    """
    The F1000XMLParserMetadata is simply a reduced version of the F1000XMLParser meant to extract
    only metadata on the paper including reviews, change notes and metadata on the publication.

    """

    def __init__(self, xml_file_path: str):
        """
        Initialize the F1000XMLParserBM for a particular paper.

        :param xml_file_path: filepath of the F1000 XML file
        """
        super(F1000XMLParserMetadata, self).__init__(xml_file_path)
        self._xml_file_path: str = xml_file_path

    def __call__(self) -> IntertextDocument:
        """
        Parse the F1000 XML document into an IntertextDocument.

        :return: the IntertextDocument
        """
        return self._parse_all_metadata()

    def _parse_all_metadata(self):
        # prep
        prefix = self._xml_file_path.split(os.path.sep)[-1]

        # use standard meta parser
        self._parse_meta()

        # add meta fields
        meta_node = self._root.find('.//article-meta')
        subs = meta_node.findall(".//article-categories/subj-group/subject")
        self._meta["subjects"] = [s.text for s in
                                  subs[2:]]  # first always = article type, not subject; second always = articles

        # add year
        year = meta_node.find(".//pub-date[@pub-type='epub']/year")
        self._meta["year"] = year.text if year is not None else None #todo verify

        # add license information
        license_node = meta_node.find(".//permissions")
        self._meta["license"] = self._stringify(license_node)

        # ** COPIED FROM PARENT CLASS __call__ method **
        # ** START **
        # Reviews
        reviews = {}
        for review in self._root.xpath('.//sub-article[@article-type="ref-report"]'):
            review_id = review.attrib['id']
            license = review.find('.//license').attrib['{http://www.w3.org/1999/xlink}href']
            recommendation = review.find('.//meta-value').text  # TODO: A bit dirty here
            doi = review.find('.//front-stub/article-id[@pub-id-type="doi"]').text
            contributors = [
                {
                    'surname': contrib.find('.//name/surname').text,
                    'given-names': contrib.find('.//name/given-names').text
                } for contrib in review.find('.//front-stub/contrib-group')
                if contrib.tag == 'contrib' and (contrib.find('.//name/surname') is not None)
            ]
            #todo get license
            meta = {'review_id': review_id,
                    'license': license,
                    'recommendation': recommendation,
                    'doi': doi,
                    'contributors': contributors}
            reviews[review_id] = self._parse(review, meta, review_id)

        # Revision comment
        version_changes = self._root.xpath('.//sec[@sec-type="version-changes"]')
        if version_changes:
            # TODO: Check if any nodes need additional parsing
            nodes, edges = self._parse_tree(version_changes[-1])
            self._add_supplementary_edges(nodes, edges)
            if prefix:
                prefix = f'revision_{prefix}'
            revision = IntertextDocument(nodes, edges, prefix or 'revision')
        else:
            revision = None
        # ** END **

        return self._meta, reviews, revision


class IntertextSentenceSplitterBM(IntertextSentenceSplitter):
    def __init__(self, itg: IntertextDocument, splitter: SentenceSplitter = None,
                 gold: Dict[str, List[Dict[str, str]]] = {}):
        super().__init__(itg, splitter, gold)

    def _get_sentences_from_itg(self) -> List[SpanNode]:

        sentence_nodes = []
        for node in self.itg.nodes:
            if node.ntype in [NTYPE_PARAGRAPH, NTYPE_ABSTRACT, NTYPE_HEADING, NTYPE_TITLE, NTYPE_LIST_ITEM]:
                boundaries = self.splitter.split(node.content)

                new_sentence_nodes = make_sentence_nodes(
                    node,
                    boundaries
                )

                sentence_nodes += new_sentence_nodes

        return sentence_nodes


class IntertextLayoutTagger:
    def __init__(self, itg, position_information):
        self.itg = itg
        self.posinfo = position_information

    def tag_text_with_position_information(self):
        out = copy.deepcopy(self.itg)
        out.meta.update({
            "position_tag_type": "from_draft"
        })

        new_root = self.itg.root
        line_nodes = []
        lines_per_node = {}
        for line in self.posinfo["lines"]:
            li, lt = line

            lt_like = lt.lower().strip()
            if lt.endswith("-"):
                lt_like = lt_like[:-1]

            pix = next(i for i,p in enumerate(self.posinfo["pages"]) if int(p[0][0]) <= int(li) <= int(p[1][0]))

            for node in self.itg._unroll_graph(new_root):
                if node.ntype in [NTYPE_PARAGRAPH, NTYPE_LIST_ITEM, NTYPE_TITLE, NTYPE_HEADING,
                                  NTYPE_ABSTRACT, NTYPE_FORMULA, NTYPE_BIB_ITEM]:
                    if lt_like in node.content.lower().strip():
                        str_idx = node.content.lower().find(lt_like)
                        new_root = node

                        posi_node = SpanNode(ntype="line",
                                             src_node=node,
                                             start=str_idx,
                                             end=str_idx + len(lt_like),
                                             meta={"created_by": "IntertextLayoutTagger", "line": li, "page": pix+1})
                        posi_node.ix = f"{node.ix}@{len(line_nodes)}_{li}"
                        line_nodes += [posi_node]

                        for nn in self.itg.breadcrumbs(node, Etype.PARENT):
                            lpn = lines_per_node.get(nn.ix, [])
                            lines_per_node[nn.ix] = [min(lpn + [li]), max(lpn + [li])]

                        break

        layout_nodes = []
        for n, lpn in lines_per_node.items():
            pix0 = next(i for i, p in enumerate(self.posinfo["pages"]) if int(p[0][0]) <= int(lpn[0]) <= int(p[1][0]))
            pix1 = next(i for i, p in enumerate(self.posinfo["pages"]) if int(p[0][0]) <= int(lpn[1]) <= int(p[1][0]))

            node = self.itg.get_node_by_ix(n)

            posi_node = SpanNode(ntype="line_range",
                                 src_node=node,
                                 start=0,
                                 end=len(node.content),
                                 meta={"created_by": "IntertextLayoutTagger",
                                       "line_start": lpn[0],
                                       "line_end":lpn[1],
                                       "page_start": pix0,
                                       "page_end": pix1})

            posi_node.ix = f"{n}@{lpn[0]}-{lpn[1]}"
            layout_nodes += [posi_node]

        # todo future: propagate the layout information to the parents until root

        for n in line_nodes + layout_nodes:
            out.add_node(n)
            edge = out.get_edge_by_ix(f'{n.src_node.ix}_{n.ix}_link')
            meta = {"created_by": "IntertextLayoutTagger"}
            if edge.meta is None:
                edge.meta = meta
            else:
                edge.meta.update(meta)

        return out


#
## Functions
#

def f1000xml_to_itg(xml_path):
    assert os.path.exists(xml_path) and os.path.isfile(xml_path), "provided file path to the XML does not exist. " \
                                                                  "Cannot parse."

    parser = F1000XMLParserBM(xml_path)
    itg_doc = parser()

    return itg_doc


def pdf_to_tei(pdf_path, config=None):
    assert os.path.exists(pdf_path) and os.path.isfile(pdf_path), "provided file path to the PDF does not exist. " \
                                                                  "Cannot parse."

    try:
        client = GrobidClient(**GROBID_CONF)
    except ServerUnavailableException as e:
        print("GROBID server not available. ERROR during pdf parsing.")
        raise e

    # use default config if none is provided
    if config is None:
        # no consolidation
        config = {
            "generateIDs": False,
            "consolidate_header": False,
            "consolidate_citations": False,
            "include_raw_citations": False,
            "include_raw_affiliations": False,
            "tei_coordinates": False,
            "segment_sentences": False
        }

    _, status, parsed = client.process_pdf("processFulltextDocument",
                                           pdf_path,
                                           **config)

    return status, parsed


def tei_to_itg(tei):
    parser = TEIXMLParser(tei)
    itg_doc = parser()

    return itg_doc
