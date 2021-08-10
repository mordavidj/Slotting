import pyodbc
import os

def connect_db(DB: str = 'Items'):
    '''Connect to a Microsoft Access database localy stored.

    '''
    string = ''

    if os.path.isdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM"):
        string = r'DBQ=C:\Users\David.Moreno\OneDrive - Visible SCM\db\{0:s}.accdb;'.format(DB)

    # Coding on my home computer
    elif os.path.isdir('D:/OneDrive - Visible SCM'):
        string = r'DBQ=D:\OneDrive - Visible SCM\db\{0:s}.accdb;'.format(DB)

    else:
        raise exception

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

