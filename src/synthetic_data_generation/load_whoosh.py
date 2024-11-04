
try:
  if qp is None: assert False
except:
  bm25_dir = "./riverbed"
  index = whoosh_index.open_dir(bm25_dir)
  searcher = index.searcher()
  qp = QueryParser("content", schema=index.schema)

