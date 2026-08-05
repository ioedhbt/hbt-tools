# tools/process/process_flow_illustration/

The HBT Process Flow Illustration page: `process_flow.py` embeds one of two
standalone, hand-written WebGL HTML files — `inp_hbt_process_flow.html`
("Formal", the full InP HBT flow) and `qad_hbt_process_flow.html`
("QAD", quick-and-dirty) — in an iframe. A segmented radio picks which.

Sibling folder [`tools/process/ebeam/`](../ebeam/AGENTS.md) holds the
unrelated EBL calculator. The two share only the `process` i18n group (the
sidebar's "Process" section) — nothing else. Unlike `ebeam/`,
**this page imports `tools.common` freely**; it is never launched standalone,
so the EBL folder's self-containment rule does not apply here.

## The one rule

Nothing here parses, regenerates or otherwise understands the HTML files —
each is read as text and handed straight to `st.iframe`. Changing the
illustration means editing the relevant `.html` file directly; nothing in
`process_flow.py` needs to change to match.

The one exception is language. Both HTML files carry their own English /
繁體中文 layer and their own toggle (so each works standalone, downloaded and
opened with no server behind it), off the same handshake line, verbatim in
both files:

```js
const HOST = {lang:null, embed:false};
```

Embedded in the portal there must be exactly one language control — the
sidebar's 🌐 — so `process_flow.py` rewrites that single line to hand over
the active language and hide the document's own toggle. If a file's
handshake line ever moves or duplicates, the page falls back to embedding it
untouched (`_hosted` returns `None`): worse cosmetics (two toggles), never a
broken page. Don't move the HTML files' translation strings into
`tools/common/i18n.py` — that would break each file working standalone,
which is the whole point of keeping them self-contained.

## Adding a third variant

Add the file here, add its path + a bilingual label to the `_VARIANTS` table
in `process_flow.py`, and make sure it carries the same `HOST` handshake line
verbatim — nothing else needs to change.
