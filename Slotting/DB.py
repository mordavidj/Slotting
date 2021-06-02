import pyodbc

def connect_db(DB: str = 'Items'):
    '''Connect to a Microsoft Access database localy stored.

    '''
    string = r'DBQ=..\..\..\db\{0:s}.accdb;'.format(DB)
    #print(string)
    try:
        connection = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};' + string)
        connection.autocommit = False
        return connection

    except pyodbc.Error as err:
        raise Exception(err)
        return -1
    
    #cursor = conn.cursor()
    #cursor.execute('select * from Customer')
   
    #for row in cursor.fetchall():
    #    print (row)

