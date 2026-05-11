from ppcgrader.info_utils import *
from ppcgrader.doc_builder import *
from ppcgrader.reporter import json_to_output

code = "is"
name = "IS"
descr = "image segmentation"


def explain_web(raw: dict):
    return generate_html(explain(json_to_output(raw), "web"))


def explain_terminal(r, color=False):
    printer = TerminalPrinter(color=color)
    return generate_term(explain(r, "term"), printer)


def explain(r, mode: str):

    fail_style = "verywrong"
    if mode == "web":
        fail_style = "strong"  #to preserve old web format

    domain = get_domain(mode=mode, domain=code)

    builder = DocumentBuilder(mode)
    input = r.input_data or {}
    output = r.output_data or {}
    oe = r.output_errors or {}
    expected = oe.get('expected') or {}
    threshold = safenum(oe.get('threshold'))

    ny = input.get('ny', None)
    nx = input.get('nx', None)
    data = input.get('data', None)

    if ny is not None and nx is not None:
        with builder.paragraph() as txt:
            txt += domain.gettext('function_called_with_dims', ny=ny, nx=nx)

        if data is not None:
            with builder.paragraph() as txt:
                txt += domain.gettext('input_data_like')

            with builder.matrix(ny, nx) as mat:
                for i in range(ny):
                    for j in range(nx):
                        mat.entry(
                            i, j,
                            TextNode([
                                StringNode(safeprint(safeget(data, i, j, 0)),
                                           style='vector'),
                                StringNode(safeprint(safeget(data, i, j, 1)),
                                           style='vector'),
                                StringNode(safeprint(safeget(data, i, j, 2)),
                                           style='vector')
                            ]), "vector")

    if oe.get('wrong_output'):
        with builder.paragraph() as txt:
            txt += domain.gettext('output_expected_comparison')

        row_count = 4 + 6  # 4 coordinate fields + 6 array fields

        with builder.table(row_count, 3, aligns=['left', 'right',
                                                 'right']) as tbl:
            tbl.header(0, '')
            tbl.header(1, domain.gettext('output_header'))
            tbl.header(2, domain.gettext('expected_header'))

            row_idx = 0
            for k in ['y0', 'x0', 'y1', 'x1']:
                a = safeget(output, k)
                b = safeget(expected, k)
                tbl.entry(row_idx, 0, k)
                if a != b:
                    tbl.entry(row_idx, 1, StringNode(safeprint(a), fail_style))
                    tbl.entry(row_idx, 2, StringNode(safeprint(b), fail_style))
                else:
                    tbl.entry(row_idx, 1, safestr(a))
                    tbl.entry(row_idx, 2, safestr(b))
                row_idx += 1

            for k in ['inner', 'outer']:
                for i in [0, 1, 2]:
                    a = safeget(output, k, i)
                    b = safeget(expected, k, i)
                    label = f'{k}[{i}]'
                    tbl.entry(row_idx, 0, label)
                    if abs(safenum(a) - safenum(b)) > threshold > 0:
                        tbl.entry(row_idx, 1,
                                  StringNode(safeprint(a), fail_style))
                        tbl.entry(row_idx, 2,
                                  StringNode(safeprint(b), fail_style))
                    else:
                        tbl.entry(row_idx, 1, safeprint(a))
                        tbl.entry(row_idx, 2, safeprint(b))
                    row_idx += 1

    return builder.build()
