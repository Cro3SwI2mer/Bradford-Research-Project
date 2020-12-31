import py_compile
import datetime

py_compile.compile('preprocessor.py')
py_compile.compile('numerical.py')
py_compile.compile('graph.py')

print("Last compilation:", str(datetime.datetime.now()))