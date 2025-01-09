import os
import re
import logging
from io import BytesIO

import PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import ContentStream, TextStringObject, NameObject, NumberObject


line_num_matcher = re.compile("^\d{3,4}$")


def clean_pdf_draft(pdf_path, with_meta_data=False):
    line_accu = 0

    def _is_line_number(operands):
        global line_num_matcher
        nonlocal line_accu

        text = operands[0][0]

        if not isinstance(text, TextStringObject):
            return False

        if not line_num_matcher.match(text):
            return False

        lnum = int(text)
        is_incremental = lnum == line_accu + 1
        is_first = lnum == 0 and line_accu == 0

        if not is_incremental and not is_first:
            return False

        line_accu = lnum
        return True

    def _is_author_information(operands):
        if len(operands) != 1:
            return False

        return "".join([str(o) for o in operands[0] if isinstance(o, TextStringObject)]) == "AnonymousACLsubmission"

    def _parse_lines(lines):
        res = []
        for line in lines:
            line_number = str(line[0][0][0])
            contents = [lii for l in line[1] for li in l for lii in li]

            parsed = ""
            for c in contents:
                if type(c) == TextStringObject:
                    parsed += str(c)
                elif type(c) == NumberObject and int(c) < 0:
                    parsed += " "

            res += [(line_number, parsed)]
        return res

    out_dir = pdf_path[:-len(pdf_path.split(os.path.sep)[-1])]
    out_path = os.path.join(out_dir, "clean.pdf")

    try:
        pdf_reader = PdfFileReader(pdf_path)
    except PyPDF2.errors.PdfReadError as err:
        if "PDF starts with" in str(err):
            logging.info("Invalid PDF header format -- seeking a point of entry")

            with open(pdf_path, "rb") as bf:
                b = bf.read(1024)
                index = b.find(b"%PDF-")
                if index == -1:
                    raise err

            with open(pdf_path, "rb") as bf:
                bf.seek(index)
                bstream = BytesIO(bf.read())

            pdf_reader = PdfFileReader(bstream)
        else:
            raise err

    pdf_writer = PdfFileWriter()

    pages = []
    lines = []
    for page in pdf_reader.pages:
        content_object = page["/Contents"].getObject()
        content = ContentStream(content_object, pdf_reader)

        line_content = []
        page_content = []

        # Loop over all pdf elements
        # source https://gist.github.com/668/2c8f936697ded94394ff4a6ffa4ae87e
        to_delete = []
        op_i = 0
        first_line = True
        for operands, operator in content.operations:
            # You might adapt this part depending on your PDF file
            # > Below the operators inferred from https://pypdf2.readthedocs.io/en/latest/_modules/PyPDF2/_page.html
            # > Check PDF iso standard for actual and full list
            #
            # Tf = text font
            # Tfs = text font size
            # Tc = character spacing
            # Th = horizontal scaling
            # Tl = leading
            # Tmode = text rendering mode
            # Trise = text rise
            # Tw = word spacing
            # Tj, ', ", TJ = Text showing [the actual text]
            # T* = text positioning
            # ...
            if operator == b"TJ":
                if _is_line_number(operands):
                    to_delete += [op_i]

                    # first line of the page -- get the last entry in line_content as a content
                    if first_line:
                        page_content += [(operands, [line_content[-1]] if len(line_content) > 0 else [])]
                        first_line = False
                    else:
                        page_content += [(operands, line_content)]

                    line_content = []
                elif _is_author_information(operands):
                    to_delete += [op_i]
                else:
                    line_content += [operands]

            op_i +=1

        # delete elements from PDF
        acc = 0
        for d in sorted(to_delete):
            del content.operations[d - acc]
            acc += 1

        # todo try setting the rendered pane to the covered subset in ACL pdfs to guarantee excluding of weird fields
        #page.mediabox.upper_right = (page.mediabox.right / 2, page.mediabox.top / 2)

        # Set the modified content as content object on the page
        page.__setitem__(NameObject('/Contents'), content)

        # parse the lines
        page_content = _parse_lines(page_content)

        # store page (start and end line) and lines
        if len(page_content) > 0:
            pages += [(page_content[0], page_content[-1])]
            lines += page_content

        # Add the page to the output
        pdf_writer.addPage(page)

    meta = {}
    meta["num_pages"] = len(pages)
    meta["num_lines"] = len(lines)
    meta["lines"] = lines
    meta["pages"] = pages

    references_line = [l for l in lines if l[1].strip() == "References"]
    if len(references_line) > 0 and len(references_line[0]) > 0:
        meta["bib_page_index"] = next(i for i, p in enumerate(pages) if int(p[0][0]) <= int(references_line[0][0]) <= int(p[1][0]))
    else:
        meta["bib_page_index"] = None

    # override document info to a default value
    pdf_writer.add_metadata({
        "author": "Anonymous",
        "author_raw": "Anonymous"
    })

    # write to file
    with open(out_path, "wb") as output_file:
        pdf_writer.write(output_file)

    # default warning for validation
    if line_accu < 500:
        logging.info(f"WARNING: Encountered a very low number of lines ({line_accu} < 500) in {output_file}.")

    return meta