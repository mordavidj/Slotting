import pyodbc

def connect_db():
    try:
        connection = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=D:\OneDrive - Visible SCM\db\Items.accdb;')
        connection.autocommit = False
        return connection

    except:
        print('DB connection failed.')
        return -1
    
    #cursor = conn.cursor()
    #cursor.execute('select * from Customer')
   
    #for row in cursor.fetchall():
    #    print (row)

