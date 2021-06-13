import sys

def funcion(req):
   parametros = req.args.split('&')
   for parametro in parametros:
      nombre,valor  = parametro.split('=')
      print (nombre,valor)

print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)
print("Hola mundo.")
