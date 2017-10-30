from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True) 
# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server
server = SimpleXMLRPCServer(("0.0.0.0", 9997),
                            requestHandler=RequestHandler)
server.register_introspection_functions()

def getvector(word):
	result = []
	try:
		result = model[word].tolist()
	except:
		print("Word "+word+"not present.")
		pass
	return result


server.register_instance(model)
server.register_function(getvector)

print("Server stared")
# Run the server's main loop
server.serve_forever()