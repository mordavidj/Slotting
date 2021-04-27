import pandas as pd
import datetime
from Slotting import *
from Item import *
from DB import *


def waldo():
    feb = pd.read_csv('data/7826_3PFShippingDetails_Feb 2021.csv',
                      header = 2,
                      na_values = 0,
                      dtype = {'Part Number': 'string',
                               'Order Number': 'string'})\
                                   .rename(columns = {'Order Number': 'order_number',
                                                      'Part Number': 'item_id',
                                                      'Quantity': 'quantity'})
    #print(feb.head())
    #print(feb.dtypes)
    #print(len(feb))
    mar = pd.read_csv('data/7826_3PFShippingDetails_March until 19.03.2021 .csv',
                      header = 2,
                      dtype = {'Part Number': 'string',
                               'Order Number': 'string'},
                      na_values = 0)\
                          .rename(columns = {'Order Number': 'order_number',
                                             'Part Number': 'item_id',
                                             'Quantity': 'quantity'})
    #print(len(mar))

    df = feb.append(mar, ignore_index = True)
    df['quantity'] = df['quantity'].fillna(0).astype(int)

    #print(len(df))

    h = generate_hashkey(df)

    #print(h)

    slotting(h, [27, 48], 'Waldo')


def level():

    weeks = 6
    pf = 48

    df = pd.read_csv('data/LeVel Optimization Hashkey.csv',
                     dtype = 'string')\
                         .rename(columns = {'Created Date': 'datetime',
                                            'Optimization Hash Key': 'hashkey'})

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['day'] = df['datetime'].dt.day

    sub = df[df.day.isin([5, 15, 25])]
    norm = df[~df.day.isin([5, 15, 25])]

    print(sub.date.value_counts())
    print(norm.date.value_counts())

    pf_norm = slotting(norm, [48], 'LeVel-normal')
    pf_sub = slotting(sub, [48], 'LeVel-sub')

    pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')

    norm_lst = pf_norm[0].list_items()
    sub_lst = pf_sub[0].list_items()
    
    #print(norm_lst)
    #print(sub_lst)

    print('\nItems in normal but not subscription:')
    for i in norm_lst:
        if i not in sub_lst:
            print(i)

    print('\nItems in subscription but not normal:')
    for i in sub_lst:
        if i not in norm_lst:
            print(i)


def level_continuous():
    hashkey = load_powerBI_hashkey('data/LeVel Optimization Hashkey.csv')

    pf2 = continuous_slotting(hashkey, [48, 32], 'LeVel_continuous')


def nuskin():
    hashkey = load_powerBI_hashkey('data/nuskin_hashkey.csv')

    #print(hashkey[['date', 'hashkey']].groupby('date').agg(['count']))

    ignored = ['01003882', '01003883', '01102892', '01003904', 
               '01310011', '01003440', '01003901', '01003529']

    pf = slotting(hashkey, [27, 27], 'NuSkin', ignore = ignored)


def truvision():
    pfs = [9, 48]
        
    hashkey = generate_hashkey_ASC('data/truvision-asc-orders.csv', 'truvision')

    print(hashkey.head())

    hashkey = hashkey.set_index('order_number')

    hashkey.to_csv('data/truvision_hashkey.csv')

    #hashkey = load_hashkey('data/truvision_hashkey.csv')

    pf = slotting(hashkey, pfs, 'TruVision')
    #pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')




def cheese(*args, **kwargs):
    print(f'args: {args}, kwargs: {kwargs}')




def kits_from_ASC(filepath, cust):
    df = pd.read_csv(filepath,
                      dtype = "string")

    dict = {}
    n_frame = []    # Stores all items in the desired format
    n_row = []      # Keeps each order together in one row
    hash = []


    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        if row['VMI_CUSTID'] == cust:
            for i in sorted(dict.keys()):
                hash.append(str(i + '*' + dict[i]))
            
            n_row.append(';'.join(hash))
            n_frame.append(n_row.copy())
            dict = {}
            hash.clear()
            
            #print(n_frame)
            n_row.clear()
            n_row.append(row['VMI_CUSTID'])
            n_row.append(row['ITEMID'])
            n_row.append(row['DESCRIPTION'])

        else:
            dict[str(row["VMI_CUSTID"])] = str(row["ITEMID"])

    for i in sorted(dict.keys()):
        hash.append(str(i + '*' + dict[i]))
            
    n_row.append(';'.join(hash))
    n_frame.append(n_row.copy())

    # first item is a blank line, so pop it
    n_frame.pop(0)
    kits = pd.DataFrame(n_frame, columns = ['customer', 'kit_id', 'description', 'hashkey'])
    kits = kits.set_index('client').sort_values('kit_id')
    kits.to_csv(f'{cust}_kits.csv')
    print('Done')
    return kits


def kits_from_ASC_to_SQL(filepath, cust):
    df = pd.read_csv(filepath,
                      dtype = "string")

    dict = {}
    kits = []    # Stores all items in the desired format
    kiq = []     # Keeps each order together in one row
    kit = ''

    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        if row['VMI_CUSTID'] == cust:
            kit = row['ITEMID']
            kits.append([row['VMI_CUSTID'], kit, row['DESCRIPTION']])

        else:
            kiq.append([kit, row['VMI_CUSTID'], row["ITEMID"]])


    # first item is a blank line, so pop it
    df_kits = pd.DataFrame(kits, columns = ['customer', 'kit_id', 'description'])
    df_kiq = pd.DataFrame(kiq, columns = ['kit', 'item', 'qty'])

    df_kits = df_kits.set_index('customer')
    df_kiq = df_kiq.set_index('kit')

    df_kits.to_csv(f'{cust}_Kit_SQL.csv')
    df_kiq.to_csv(f'{cust}_Kits_Items_SQL.csv')
    print('Done')
    return kits

def main():
    #level_continuous()
    #waldo()
    #min_max_from_hashkey('do')
    #nuskin()
    truvision()   
    #kits_from_ASC_to_SQL("C:\\Users\\David.Moreno\\OneDrive - Visible SCM\\Desktop\\truvision_kit.csv", 'TRUVISION') 
    
def test():

    cnxn = connect_db()
    cursor = cnxn.cursor()

    sql = '''SELECT item, item_qty FROM Kits_Items 
             WHERE kit = '3467';'''

    d = cursor.execute(sql).fetchall()

    #d = cursor.fetchall()
    print(d)

    for r in d:
        print(r[0])

    cnxn.close()

main()
#test()
