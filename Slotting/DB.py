import pyodbc

def connect_db():
    try:
        connection = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=..\..\..\db\Items.accdb;')
        connection.autocommit = False
        return connection

    except pyodbc.Error as err:
        raise Exception(err)
        return -1
    
    #cursor = conn.cursor()
    #cursor.execute('select * from Customer')
   
    #for row in cursor.fetchall():
    #    print (row)

