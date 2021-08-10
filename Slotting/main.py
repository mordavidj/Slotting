import pandas as pd
pd.set_option('max_columns', None)
#pd.set_option('max_rows', None)
import math
import csv
import datetime as dt
import os
import copy
import matplotlib.pyplot as plt
import pyodbc
import numpy as np
import multiprocessing as mp
from Slotting import *
from Hashkey import *

MAX_PROCESSORS = 6

POD = [1, 3, 3]
HVPNP = [3, 3, 3]
PNP = [4, 3, 4]

MIN_LOL_ORDERS = 50
MAX_LOL_LINES = 3
MAX_LOL_ITEMS = 5

ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


POD = [1, 3, 3]
HVPNP = [3, 3, 3]
PNP = [4, 3, 4]


def test():
    df_wj = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\YoungLiving_WJ_OHK.xlsx")
    df_me = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\YoungLiving_ME_OHK.xlsx")
    df_lu = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\LUME_OHK.xlsx")

    hashkey_wj = df_wj[(df_wj['date'] >= dt.date(year=2021, month=6, day=27)) & (df_wj['date'] <= dt.date(year=2021, month=7, day=3))]
    hashkey_me = df_me[(df_me['date'] >= dt.date(year=2021, month=6, day=27)) & (df_me['date'] <= dt.date(year=2021, month=7, day=3))]
    hashkey_lu = df_lu[(df_lu['date'] >= dt.date(year=2021, month=6, day=27)) & (df_lu['date'] <= dt.date(year=2021, month=7, day=3))]
    
    print(hashkey_wj)
    print(hashkey_me)
    print(hashkey_lu)

    single_single_analyze(hashkey_wj)
    single_single_analyze(hashkey_me)
    single_single_analyze(hashkey_lu)

#test()

def test1():
    hashkey = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\MANSCAPED_hashkey.csv")

    pf_mk = Pickface()
    pf_mk.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\MANSCAPED-MK.csv")
    pf_mh = Pickface()
    pf_mh.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\MANSCAPED-MH.csv")

    evaluate([pf_mk, pf_mh], hashkey, True)
    
#test1()

def test2():
    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\xyngular_WJ_orders.csv", 'XYNGULAR')
    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\xyngular_ATL_orders.csv", 'XYNGULAR')

    #hash_pura_wj = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\PURA_WJ_hashkey.csv")
    hash_pura_atl = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\PURA_ATL_hashkey.csv")
    #print(hashkey)
    #single_single_analyze(hashkey)
    pf_pura = Pickface()
    pf_pura.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\PURA_ATL_24_PRA.csv")
    evaluate([pf_pura], hash_pura_atl, True)

    hash_xyng_wj = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\XYNGULAR_WJ_hashkey.csv")
    hash_xyng_atl = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\XYNGULAR_ATL_hashkey.csv")

    pf_xyng_wj = Pickface()
    pf_xyng_wj.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\XYNGULAR_WJ_27.csv")
    pf_xyng_atl_pod = Pickface()
    pf_xyng_atl_pod.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\XYNGULAR_ATL_POD_XYF.csv")
    pf_xyng_atl_54 = Pickface()
    pf_xyng_atl_54.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\XYNGULAR_ATL_54_XYN.csv")

    evaluate([pf_xyng_wj], hash_xyng_wj, True)
    evaluate([pf_xyng_atl_pod, pf_xyng_atl_54], hash_xyng_atl, True)

#test2()

def test3():
    hashkey = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\TRUVISION_hashkey.csv")
    pf_48 = Pickface()
    pf_48.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\TruVision-48.csv")
    pf_pod = Pickface()
    pf_pod.from_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\TruVision-9.csv")

    evaluate([pf_48, pf_pod], hashkey, True)

#test3()

def test4():
    df_frame = []
    df_pf_info = []
    df_items = []

    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\doterra_ATL_orders.csv", 'DOTERRA')

    slotting(pd.Series({'client': 'BODYGUARDZ', 
                        'hashkey': r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\hashkey\BODYGUARDZ_hashkey.csv", 
                        'lol': 'FALSE', 
                        'single': 'FALSE', 
                        'pfs': '1,4,25', 
                        'heights': '99,99,99,99', 
                        'required': None, 
                        'ignored': None}, 
                       index = ['client', 'hashkey', 'lol', 'single', 'pfs', 'heights', 'required', 'ignored']), 
             df_frame, df_pf_info, df_items)

    slotting(pd.Series({'client': 'BODYGUARDZ', 
                        'hashkey': r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\hashkey\BODYGUARDZ_hashkey.csv", 
                        'lol': 'FALSE', 
                        'single': 'FALSE', 
                        'pfs': '1,4,50', 
                        'heights': '99,99,99,99', 
                        'required': None, 
                        'ignored': None}, 
                       index = ['client', 'hashkey', 'lol', 'single', 'pfs', 'heights', 'required', 'ignored']), 
             df_frame, df_pf_info, df_items)

    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Type', 'Order', 'Percent', 'strPercent', 'Items'])
    print(pfs_info)

    columns = ['client', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max']
    pfs = pd.DataFrame(df_frame, columns = columns)

    items = pd.DataFrame(df_items, columns = ['Client', 'Pickface', 'Item', 'Percent', 'Average'])

    print(pfs)
    pfs.to_csv('pfs_BODYGUARDZ.csv')
    pfs_info.to_csv('pfs_info_BODYGUARDZ.csv')

#test4()

def test5():
    hashkey = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\hashkey\BODYGUARDZ_hashkey.csv")
    #hashkey = remove_lol(hashkey)

    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    print(df)

    cnxn = connect_db()

    #print(order_count)
    #print(len(order_count[order_count.order_count == 1]))

    # Connect to DB to get item info
    item_sql = '''SELECT i.ASC_id AS item_id, i.description
                    FROM Item AS i
                    WHERE i.customer = ucase('{0:s}');'''.format('bodyguardz')

    item_info = pd.read_sql(item_sql, cnxn).set_index('item_id')

    print(item_info)

    cnxn.close()
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    print(df)

    # Pivot so that day sums of all item quantities are aligned on the same row with the date
    pivot_df = df.pivot(index = 'date', columns = 'item_id', values = 'qty')
    min_max = pd.DataFrame(columns = ['item_id', 'mean', 'median', 'max'])
    pivot_df = pivot_df.fillna(0)
    #print(pivot_df)

    for i in range(0, len(pivot_df.columns)):
        min_max = min_max.append({'item_id': str(pivot_df.columns[i]),
                                  'mean': pivot_df[pivot_df.columns[i]].mean(), 
                                  'median': pivot_df[pivot_df.columns[i]].median(),
                                  'max': pivot_df[pivot_df.columns[i]].max()}, 
                                 ignore_index = True)

    min_max = min_max.set_index('item_id').join(item_info, how='left')

    min_max.to_excel('BODYGUARDZ_item_velocity.xlsx')

#test5()

def test6():
    hashkey = load_hashkey(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA\LEVEL_hashkey.xlsx")
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    #print(df)
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    print(df)
    df.to_excel('level_item_velocity.xlsx')

#test6()

def test7():
    df_frame = []
    df_pf_info = []
    df_items = []

    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\doterra_ATL_orders.csv", 'DOTERRA')

    slotting(pd.Series({'client': 'MANSCAPED', 
                        'hashkey': "hashkey/MANSCAPED_hashkey.csv", 
                        'lol': 'TRUE', 
                        'single': 'FALSE', 
                        'pfs': '2,3,3;HVPnP', 
                        'heights': '99,99,99;99,99,99', 
                        'required': None, 
                        'ignored': 'MANKPP2L;MANKPP2M;MANKPP2S;MANKPP2XL;MANKPP2XXL;MANKPP3L;MANKPP3M;MANKPP3S;MANKPP3XL;MANKPP3XXL;MANKPP3XXXL;MANKPP4L;MANKPP4M;MANKPP4S;MANKPP4XL;MANKPP4XXL;MANKPP4XXXL;PHP-LD;PHP-LM;PHP-LP;PHP-LR;PHP-LW;PHP-WD;PHP-WM;PHP-WP;PHP-WR'}, 
                       index = ['client', 'hashkey', 'lol', 'single', 'pfs', 'heights', 'required', 'ignored']), 
             df_frame, df_pf_info, df_items)

    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Type', 'Order', 'Percent', 'strPercent', 'Items'])
    print(pfs_info)

    columns = ['client', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)

    items = pd.DataFrame(df_items, columns = ['Client', 'Pickface', 'Item', 'Orders', 'Quantity'])

    print(pfs)
    print(items)
    pfs.to_excel('TEST_pfs_MANSCAPED.xlsx')
    pfs_info.to_excel('TEST_pfs_info_MANSCAPED.xlsx')

#test7()

def test8():
    df_frame = []
    df_pf_info = []
    df_items = []

    slotting(pd.Series({'client': 'LIFEVAN', 
                        'hashkey': 'hashkey/LIFEVAN_hashkey.xlsx', 
                        'lol': 'TRUE', 
                        'single': 'FALSE', 
                        'pfs': '1,3,6', 
                        'heights': '99,99,99', 
                        'required': None, 
                        'ignored': None}, 
                       index = ['client', 'hashkey', 'lol', 'single', 'pfs', 'heights', 'required', 'ignored']), 
             df_frame, df_pf_info, df_items)

    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Type', 'Order', 'Percent', 'strPercent', 'Items'])
    print(pfs_info)

    columns = ['client', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)

    items = pd.DataFrame(df_items, columns = ['Client', 'Pickface', 'Item', 'Percent', 'Average'])

    print(pfs)

    with pd.ExcelWriter('TEST_LIFEVAN_pfs.xlsx') as writer:
        pfs_info.set_index('Client')\
            .to_excel(writer, sheet_name = 'Summary')
        pfs.set_index('client')\
            .to_excel(writer, sheet_name = 'PFS')
        items.set_index('Client')\
            .to_excel(writer, sheet_name = 'LOL & Cart')


def test9():
    hashkey = load_hashkey('hashkey/LIFEVAN_hashkey.xlsx')
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    #print(df)
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    print(df)
    df.to_excel('lifevan_item_velocity.xlsx')


def test10():
    load_hashkey('hashkey/MONAT_hashkey.xlsx')

    hashkey = load_hashkey('hashkey/WALDO_hashkey.xlsx')

    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    #print(df)
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    print(df)
    df.to_excel('waldo_velocity.xlsx')


def test11():
    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\truvision_asc_orders_and_quantities.csv", client = 'TRUVISION')

    hashkey = load_hashkey(client = 'TRUVISION', period = 60)
    print(hashkey)
    pfs = pd.read_csv(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\Truvision_pfs.csv")
    
    pfs['pf'] = pfs.apply(lambda row: row['LOCATIONID'].split('.')[0], axis = 1)

    
    its = []

    for i in pfs.pf.unique():
        its.append(pfs[pfs.pf == i]['ITEMID'].tolist())

    print(its)
    lol = True

    print('\nEvaluating pickfaces')
    ord_sum = len(hashkey)

    print('\nTotal Orders: {0:,}'.format(ord_sum))
    by_date = (hashkey['date'].dt.date).value_counts().reset_index()\
        .rename(columns = {'date': 'count', 'index': 'date'})\
        .sort_values('date').set_index('date')

    by_date.plot(kind = 'hist', legend = None)
    by_date.plot(kind = 'bar', legend = None)
    plt.show()
    if lol:
        hashkey = remove_lol(hashkey)

    by_date = (hashkey['date'].dt.date).value_counts().reset_index()\
        .rename(columns = {'date': 'count', 'index': 'date'})\
        .sort_values('date').set_index('date')

    by_date.plot(kind = 'hist', legend = None)
    by_date.plot(kind = 'bar', legend = None)
    plt.show()
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False
    print(order_count)

    for i in its:
        items = pd.DataFrame(i, columns = ['item_id'])\
            .set_index('item_id')
        items['orders'] = 0
        #print(items)

        
        for index, row in order_count.iterrows():
            if all(x in items.index.tolist() for x in index.split(';')):
                order_count.at[index, 'visited'] = True
                for o in index.split(';'):
                    items.at[o, 'orders'] += row['order_count']
                #print('')
                #print(f'{index}: {row["order_count"]}')
                #print(items)
                #pass

        
        print(items)
        ord_serv = order_count[order_count.visited].order_count.sum()

        ord_per = ord_serv / ord_sum
        print('\nOrders Served by PF: {0:,} ({1:.2%})'.format(ord_serv, 
                                                              ord_per))

        visited = list(order_count[order_count.visited == True].index)
        if visited:
            sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
            sub_val_count = (sub_hashkey['date'].dt.date).value_counts()

            min = int(round(sub_val_count.min()))
            q1 = int(round(np.nanpercentile(sub_val_count, 25)))
            med = int(round(sub_val_count.median()))
            q3 = int(round(np.nanpercentile(sub_val_count, 75)))
            max = int(round(sub_val_count.max()))

            print('\tOrders/Day:')
            print(f'\tMin = {min:,}')
            print(f'\t1Qt = {q1:,}')
            print(f'\tMed = {med:,}')
            print(f'\t3Qt = {q3:,}')
            print(f'\tMax = {max:,}')

            #by_date = sub_val_count.to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
            #by_date.plot(kind = 'bar', legend = None)
            #plt.show()
            order_count = order_count[~order_count.visited]

       
        

    rem_sum = order_count.order_count.sum()
    rem_per = rem_sum / ord_sum
    print('\nOrders Remaining: {0:,} ({1:.2%})'.format(rem_sum, rem_per))


def test12():
    df_pf_info = []
    df_frame = []
    df_items = []

    # Coding on my work computer
    if os.path.isdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM"):
        os.chdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA")

    # Coding on my home computer
    elif os.path.isdir('D:/OneDrive - Visible SCM'):
        os.chdir('D:/OneDrive - Visible SCM/Coding/DATA')

    else:
        raise "Computer not recognized"

    print('Loading all client PF info . . . ', end = '')

    custs_info = pd.read_excel("truvision_analyze.xlsx",
                                dtype = 'string')
    print('Done')

    print(custs_info)

    processes = []

    for _, row in custs_info.iterrows():
        slotting(row, df_frame, df_pf_info, df_items)

    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Type', 'Order', 
                                                   'Percent', 'strPercent', 
                                                   'Items'])
    print(pfs_info)

    columns = ['client', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)
    pfs.location = pfs.location.astype('str')
    print(pfs)

    items = pd.DataFrame(df_items, columns = ['Client', 'Type', 'Item', 
                                              'Orders', 'Average'])

    pfs_info = pfs_info.set_index('Client')
    pfs = pfs.set_index('client')
    items = items.set_index('Client')

    # Save this info to be loaded on the next run
    with pd.ExcelWriter('truvision_pfs.xlsx') as writer:
        pfs_info.to_excel(writer, sheet_name = 'Summary')
        pfs.to_excel(writer, sheet_name = 'PFS')
        items.to_excel(writer, sheet_name = 'LOL & Cart')

#test12()

def test13():
    upload_orders(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\doterra_atl_orders.txt", 'ATL', 'DOTERRA')
    


def test14():
    df = pd.read_excel(r"C:\Users\David.Moreno\Downloads\Copy of Kendo_Order_SKU_Hash_20210601_ToDate_20210728.xlsx")
    print(df)
    print(df.dtypes)
    df['order_config'] = df['CONCAT_SKU_LIST'].str.replace('~',';')
    df['hashkey'] = df['CONCAT_SKU_QTY_LIST'].str.replace('~',';')
    df['hashkey'] = df['hashkey'].str.replace('(','*')
    df['hashkey'] = df['hashkey'].str.replace(')', '')
    print(df[['order_config', 'hashkey']])
    df = df.rename(columns={'SALESID':'order_number',
                            'DATAAREAID':'client'})
    df['order_date'] = pd.to_datetime(df['CreateDT_OO'])
    df1 = df[['order_number','client','order_date','hashkey','order_config']].set_index('order_number')
    print(df1)
    df1.to_excel('Kendo_hashkey.xlsx')


def test15():
    fnty = pd.read_excel('Kendo_hashkey.xlsx', sheet_name='Sheet2',dtype='string')
    fnty['date'] = pd.to_datetime(fnty['order_date'])
    print(fnty.dtypes)
    print(fnty)

    df_frame = []
    df_pf_info = []
    df_items = []

    #generate_hashkey_ASC(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\doterra_ATL_orders.csv", 'DOTERRA')

    slotting(pd.Series({'client': 'FNTY1',
                        'warehouse': 'CIN',
                        'hashkey': fnty, 
                        'lol': 'FALSE', 
                        'lol_ignore': None,
                        'single': 'FALSE', 
                        'single_ignore': None,
                        'pfs': '1,5,5', 
                        'pfs_heights': '99,99,99,99,99', 
                        'pfs_require': None, 
                        'pfs_ignore': None}, 
                       index = ['client', 'warehouse','hashkey', 'lol', 'lol_ignore','single','single_ignore','pfs', 'pfs_heights', 'pfs_require', 'pfs_ignore']), 
             df_frame, df_pf_info, df_items)

    slotting(pd.Series({'client': 'FNTY2',
                        'warehouse': 'CIN',
                        'hashkey': fnty, 
                        'lol': 'FALSE', 
                        'lol_ignore': None,
                        'single': 'FALSE', 
                        'single_ignore': None, 
                        'pfs': '2,5,5', 
                        'pfs_heights': '99,99,99,99,99', 
                        'pfs_require': None, 
                        'pfs_ignore': None}, 
                       index = ['client', 'warehouse','hashkey', 'lol', 'lol_ignore','single','single_ignore','pfs', 'pfs_heights', 'pfs_require', 'pfs_ignore']), 
             df_frame, df_pf_info, df_items)

    slotting(pd.Series({'client': 'FNTY3',
                        'warehouse': 'CIN',
                        'hashkey': fnty, 
                        'lol': 'FALSE', 
                        'lol_ignore': None,
                        'single': 'FALSE', 
                        'single_ignore': None, 
                        'pfs': '3,5,5', 
                        'pfs_heights': '99,99,99,99,99', 
                        'pfs_require': None, 
                        'pfs_ignore': None}, 
                       index = ['client', 'warehouse','hashkey', 'lol', 'lol_ignore','single','single_ignore','pfs', 'pfs_heights', 'pfs_require', 'pfs_ignore']), 
             df_frame, df_pf_info, df_items)

    slotting(pd.Series({'client': 'FNTY4',
                        'warehouse': 'CIN',
                        'hashkey': fnty, 
                        'lol': 'FALSE', 
                        'lol_ignore': None,
                        'single': 'FALSE', 
                        'single_ignore': None, 
                        'pfs': '4,5,5', 
                        'pfs_heights': '99,99,99,99,99', 
                        'pfs_require': None, 
                        'pfs_ignore': None}, 
                       index = ['client', 'warehouse','hashkey', 'lol', 'lol_ignore','single','single_ignore','pfs', 'pfs_heights', 'pfs_require', 'pfs_ignore']), 
             df_frame, df_pf_info, df_items)

    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Warehosue', 'Type', 'Order', 
                                                   'Percent', 'strPercent', 
                                                   'Items'])
    print(pfs_info)

    columns = ['client', 'warehouse', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)
    pfs.location = pfs.location.astype('str')
    print(pfs)

    items = pd.DataFrame(df_items, columns = ['Client', 'Warehouse', 'Type', 'Item', 
                                              'Orders', 'Average'])

    pfs_info = pfs_info.set_index('Client')
    pfs = pfs.set_index('client')
    items = items.set_index('Client')

    # Save this info to be loaded on the next run
    with pd.ExcelWriter('FNTY_pfs.xlsx') as writer:
        pfs_info.to_excel(writer, sheet_name = 'Summary')
        pfs.to_excel(writer, sheet_name = 'PFS')
        items.to_excel(writer, sheet_name = 'LOL & Cart')

def test16():
    df = pd.read_csv('hashkey/DOTERRA_ATL_hashkey.csv')
    df['len'] = df['hashkey'].map(lambda x: len(x))
    print(df['len'].max())
    upload_orders('hashkey/DOTERRA_ATL_hashkey.csv', 'ATL', 'DOTERRA')
    return

def linear():

    df_pf_info = []
    df_frame = []
    df_items = []

    # Coding on my work computer
    if os.path.isdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM"):
        os.chdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA")

    # Coding on my home computer
    elif os.path.isdir('D:/OneDrive - Visible SCM'):
        os.chdir('D:/OneDrive - Visible SCM/Coding/DATA')

    else:
        raise "Computer not recognized"

    test13()


    old_pfs_info = None
    old_pfs = None

    if os.path.isfile('latest_pfs.xlsx'):
        print('Loading prior slotting info . . . ', end = '')
        old_pfs_info = pd.read_excel(r"latest_pfs.xlsx", 
                                     sheet_name = 'Summary', 
                                     index_col=[0, 1])
        old_pfs = pd.read_excel(r"latest_pfs.xlsx", 
                                sheet_name = 'PFS',
                                index_col=[0, 1, 2])
    print('Done')

    print('Loading all client PF info . . . ', end = '')

    custs_info = pd.read_excel("custs_info.xlsx",
                                dtype = 'string')
    print('Done')

    print(custs_info)

    processes = []

    for _, row in custs_info.iterrows():
        slotting(row, df_frame, df_pf_info, df_items, old_pfs_info, old_pfs)


    pfs_info = pd.DataFrame(df_pf_info, columns = ['Client', 'Warehouse', 'Type', 'Order', 
                                                   'Percent', 'strPercent', 
                                                   'Items'])
    print(pfs_info)

    columns = ['client', 'warehouse', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
                'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)
    pfs.location = pfs.location.astype('str')
    print(pfs)

    items = pd.DataFrame(df_items, columns = ['Client', 'Warehouse', 'Type', 'Item', 
                                              'Orders', 'Average'])

    pfs_info = pfs_info.set_index('Client')
    pfs = pfs.set_index('client')
    items = items.set_index('Client')

    # Save this info to be loaded on the next run
    with pd.ExcelWriter('latest_pfs.xlsx') as writer:
        pfs_info.to_excel(writer, sheet_name = 'Summary')
        pfs.to_excel(writer, sheet_name = 'PFS')
        items.to_excel(writer, sheet_name = 'LOL & Cart')


    # Save the last runs into the history
    today = str(dt.date.today())

    if not os.path.isdir(f'history/{today}'):
        os.makedirs(f'history/{today}')
        with pd.ExcelWriter(f'history/{today}/pfs.xlsx') as writer:
            pfs_info.to_excel(writer, sheet_name = 'Summary')
            pfs.to_excel(writer, sheet_name = 'PFS')
            items.to_excel(writer, sheet_name = 'LOL & Cart')

    # If the folder already exists, make a new one
    else:
        i = 1
        while os.path.isdir(f'history/{today}({i})'):
            i += 1

        os.makedirs(f'history/{today}({i})')
        with pd.ExcelWriter(f'history/{today}({i})/pfs.xlsx') as writer:
            pfs_info.to_excel(writer, sheet_name = 'Summary')
            pfs.to_excel(writer, sheet_name = 'PFS')
            items.to_excel(writer, sheet_name = 'LOL & Cart')

    return


def main():

    m = mp.Manager()

    m_df_pf_info = m.list()
    m_df_frame = m.list()
    m_df_items = m.list()

    # Coding on my work computer
    if os.path.isdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM"):
        os.chdir(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\DATA")

    # Coding on my home computer
    elif os.path.isdir('D:/OneDrive - Visible SCM'):
        os.chdir('D:/OneDrive - Visible SCM/Coding/DATA')

    else:
        raise Exception("Computer not recognized")

    #test13()

    old_pfs_info = None
    old_pfs = None

    if os.path.isfile('latest_pfs.xlsx'):
        print('Loading prior slotting info . . . ', end = '')
        old_pfs_info = pd.read_excel(r"latest_pfs.xlsx", 
                                     sheet_name = 'Summary', 
                                     index_col=[0, 1, 2])
        old_pfs = pd.read_excel(r"latest_pfs.xlsx", 
                                sheet_name = 'PFS',
                                index_col=[0, 1, 2, 3])
    print('Done')

    print('Loading all client PF info . . . ', end = '')

    custs_info = pd.read_excel("custs_info.xlsx",
                                dtype = 'string')
    print('Done')

    print(custs_info)

    warehouses = custs_info['warehouse'].value_counts().to_frame()\
        .rename(columns={'warehouse':'client_count'})
    warehouses.index.names = ['warehouse']

    processes = []

    with mp.Pool(MAX_PROCESSORS) as pool:
        pool.starmap(slotting, 
                     [(row, m_df_frame, m_df_pf_info, m_df_items, old_pfs_info, 
                       old_pfs) for _, row in custs_info.iterrows()])
        pool.close()


    df_frame = list(m_df_frame)
    df_pf_info = list(m_df_pf_info)
    df_items = list(m_df_items)

    pfs_info = pd.DataFrame(df_pf_info, 
                            columns = ['Client', 'Warehouse', 'Type', 'Order', 
                                       'Percent', 'strPercent', 'Items'])
    print(pfs_info)

    columns = ['client', 'warehouse', 'pickface', 'location', 'item_id', 
               'desc', 'bay', 'row', 'col', 'min', 'max', 'change']
    pfs = pd.DataFrame(df_frame, columns = columns)
    pfs.location = pfs.location.astype('str')
    print(pfs)

    items = pd.DataFrame(df_items, 
                         columns = ['Client', 'Warehouse', 'Type', 'Item', 
                                    'Orders', 'Average'])

    pfs_info = pfs_info.set_index('Client')
    pfs = pfs.set_index('client')
    items = items.set_index('Client')

    # Save this info to be loaded on the next run
    with pd.ExcelWriter('latest_pfs.xlsx') as writer:
        pfs_info.to_excel(writer, sheet_name = 'Summary')
        pfs.to_excel(writer, sheet_name = 'PFS')
        items.to_excel(writer, sheet_name = 'LOL & Cart')
        warehouses.to_excel(writer, sheet_name = 'Warehouse')


    # Save the last runs into the history
    today = str(dt.date.today())

    if not os.path.isdir(f'history/{today}'):
        os.makedirs(f'history/{today}')
        with pd.ExcelWriter(f'history/{today}/pfs.xlsx') as writer:
            pfs_info.to_excel(writer, sheet_name = 'Summary')
            pfs.to_excel(writer, sheet_name = 'PFS')
            items.to_excel(writer, sheet_name = 'LOL & Cart')
            warehouses.to_excel(writer, sheet_name = 'Warehouse')

    # If the folder already exists, make a new one
    else:
        i = 1
        while os.path.isdir(f'history/{today}({i})'):
            i += 1

        os.makedirs(f'history/{today}({i})')
        with pd.ExcelWriter(f'history/{today}({i})/pfs.xlsx') as writer:
            pfs_info.to_excel(writer, sheet_name = 'Summary')
            pfs.to_excel(writer, sheet_name = 'PFS')
            items.to_excel(writer, sheet_name = 'LOL & Cart')
            warehouses.to_excel(writer, sheet_name = 'Warehouse')


if __name__ == "__main__":
    linear()
    main()

