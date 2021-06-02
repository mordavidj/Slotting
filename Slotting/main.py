import pandas as pd
pd.set_option('max_columns', None)
import datetime
from Slotting import *
from Pickface import *
from Item import *
from DB import *
from Hashkey import *
from GUI import *
from test import tk_tutorial



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

    df = load_powerBI_hashkey('data/LeVel Optimization Hashkey.csv')
    print(df)
    df['datetime'] = pd.to_datetime(df['date'])
    df['day'] = df['datetime'].dt.day

    sub = df[df.day.isin([5, 15, 25])]
    norm = df[~df.day.isin([5, 15, 25])]

    print(sub.date.value_counts())
    print(norm.date.value_counts())

    pf_norm = slotting(norm, [48], 'LeVel-normal', [15, 15, 40])
    pf_sub = slotting(sub, [48], 'LeVel-sub', [15, 15, 40])

    #pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')

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
    #hashkey = load_powerBI_hashkey('data/nuskin_hashkey.csv')

    #print(hashkey[['date', 'hashkey']].groupby('date').agg(['count']))

    ignored = ['01003882', '01003883', '01102892', '01003904', 
               '01310011', '01003440', '01003901', '01003529']

    #pf = slotting(hashkey, [27], 'NuSkin', ignore = ignored)

    #pf1 = Pickface()
    #pf1.from_csv(r"..\..\..\Desktop\Nuskin-Memphis-27.csv")
    #evaluate_pf(hashkey, pf1)

    df = pd.read_csv('../../../Desktop/Nuskin_hashkey_lookup.csv',
                     dtype = 'string')\
                         .rename(columns = {'Created Date': 'datetime',
                                            'Optimization Hash Key': 'hashkey'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['day'] = df['datetime'].dt.day
    print(len(df))
    
    df = df[df['day'] >= 9]
    df['item_count'] = 0
    tot = len(df)
    print(tot)

    for index, row in df.iterrows():
        hashkey = row['hashkey'].split(';')[:-1]
        i_count: int = 0
        for hash in hashkey:
            i_count += int(hash.split('*')[-1])
        df.at[index, 'item_count'] = i_count

    over = df[df.item_count > 12]

    print(len(over))
    print(len(over) / tot)
    max = int(round(over.item_count.max()))
    print(f'Max = {max:,}')


 
def epicure():
    kits_from_ASC_to_SQL(r"..\..\..\Desktop\epicure_kit_BOM.csv")
    pfs = [48]

    #hashkey = generate_hashkey_ASC(r'..\..\..\Desktop\epicure_orders_and_quantities.csv' , 'Epicure')

    #hashkey = hashkey.set_index('order_number')

    #hashkey.to_csv('data/epicure_hashkey.csv')

    hashkey = load_hashkey('data/epicure_hashkey.csv')

    pf = slotting(hashkey, pfs, 'Epicure', [15, 15, 15])


def truvision():
    pfs = [[1, 4, 5], [4, 3, 4]]
    height = [[15, 15, 15, 99], [15, 14, 16]]
    prio = [[2, 1, 0, 3], [1, 0, 2]]
        
    #hashkey = generate_hashkey_ASC(r"..\..\..\Desktop\truvision_asc_orders_and_quantities.csv", 'truvision')

    #print(hashkey.head())

    #hashkey = hashkey.set_index('order_number')

    #hashkey.to_csv('data/truvision_hashkey.csv')

    hashkey = load_hashkey('data/truvision_hashkey.csv')

    #top_vel = build_by_velocity(hashkey, pfs[0])

    #Difference of ~300 orders (0.3%) more when built by velocity and not order configuration

    required = ['PROP65']

    #pf = slotting(hashkey, pfs, 'TRUVISION', height, prio, require = required)
    
    #print(hashkey['date'].value_counts().sort_index())
    #pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')



def lifevan():
    #kits_from_ASC_to_SQL(r"..\..\..\Desktop\lifevan_kit_BOM.csv")
    #hashkey = generate_hashkey_ASC('../../../Desktop/lifevan_asc.csv', 'LIFEVAN')
    #hashkey = hashkey.set_index('order_number')
    #hashkey.to_csv('data/lifevan_hashkey.csv')
    hashkey = load_hashkey('data/lifevan_hashkey.csv')
    print(hashkey)
    nums = [27, 48]
    pf = slotting(hashkey, nums, 'LIFEVAN', [20, 20, 20])



def manscaped():
    kits_from_ASC_to_SQL(r"..\..\..\Desktop\manscaped_kit_BOM.csv")



def amare():
    kits_from_ASC_to_SQL(r"..\..\..\Desktop\amare_kit_BOM.csv")



def bodyguardz():
    #kits_from_ASC_to_SQL(r"..\..\..\Desktop\bodyguardz_kit_BOM.csv")
    #hashkey = generate_hashkey_ASC(r"..\..\..\Desktop\batch_1834971.csv", 
    #                               'BODYGUARDZ')
    #hashkey.to_csv('data/bodyguardz_hashkey.csv')
    #print(hashkey)

    #hashkey = load_hashkey('data/bodyguardz_hashkey.csv')
    single_single(r"..\..\..\Desktop\batch_1834975.csv")




def mfgdot():
    #kits_from_ASC_to_SQL(r"..\..\..\Desktop\mfgdot_kit_BOM.csv")
    hashkey = generate_hashkey_ASC(r"..\..\..\Desktop\doterra_orders_and_quantities.csv", 
                                   'MFGDOT')
    print(hashkey)
    hashkey.to_csv('data/mfgdot_hashkey.csv')
    
    hashkey = load_hashkey('data/mfgdot_hashkey.csv')
    top_vel = build_by_velocity(hashkey, 48)



def young_living():
    hashkey = load_powerBI_hashkey(r"..\..\..\Desktop\younglivingorders.csv")
    print(hashkey.date.max())
    print(hashkey.date.min())
    new_pf = Pickface()
    new_pf.from_csv(r"data\YoungLiving-28.csv")
    new_pf.display()
    new_pf.evaluate(hashkey)

    pf = slotting(hashkey, [27], 'YOUNGLIVING', [15,15,15])
    



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
    kits.to_csv(f'data/{cust}_kits.csv')
    print('Done')

    return kits



def kits_from_ASC_to_SQL(filepath):
    print(f'Getting kits from ASC for SQL . . . ', end = '')

    df = pd.read_csv(filepath,
                      dtype = "string")

    dict = {}   # Associate items and quantities
    kits = []   # Stores all items in the desired format
    kiq = []    # Keeps each order together in one row
    kit = ''    # Store the kit id while adding items
    cust = ''   # Store the customer id while adding items

    # Connect to DB to get item info
    cnxn = connect_db()
    if type(cnxn) is int:
        return

    cust_sql = '''SELECT customer_id
                  FROM Customer;'''

    cust_df = pd.read_sql(cust_sql, cnxn)

    custs = list(cust_df['customer_id'])
    

    for index, row in df.iterrows():

        if pd.notna(row['VMI_CUSTID']):
            # If the it item can be converted into a date, it's a new item
            if row['VMI_CUSTID'].upper() in custs:
                cust = row['VMI_CUSTID']
                kit = row['ITEMID']
                kits.append([cust, kit, row['DESCRIPTION']])

            else:
                kiq.append([cust, kit, row['VMI_CUSTID'], row["ITEMID"]])


    # save the items into dataframes
    df_kits = pd.DataFrame(kits, columns = ['customer', 'asc_id', 'description'])
    df_kiq = pd.DataFrame(kiq, columns = ['customer', 'kit', 'item', 'qty'])

    df_kits = df_kits.set_index('customer')
    df_kiq = df_kiq.set_index('customer')

    df_kits.to_csv(f'data/ASC_Kit_SQL.csv')
    df_kiq.to_csv(f'data/ASC_Kits_Items_SQL.csv')

    print('Done')



def main():
    #level()
    #level_continuous()
    #waldo()
    #min_max_from_hashkey('do')
    #nuskin()
    #epicure()
    truvision()   
    #manscaped() 
    #amare()
    #bodyguardz()
    #lifevan()
    #mfgdot()
    #young_living()

    #kits_from_ASC_to_SQL(r"..\..\..\Desktop\customer_kit_BOM.csv")
    pass



main()
#gui()
#tk_tutorial()

