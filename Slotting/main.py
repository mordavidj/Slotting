import pandas as pd
import datetime
from Slotting import *

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

    pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')

def nuskin():
    hashkey = load_powerBI_hashkey('data/nuskin_hashkey.csv')

    #print(hashkey[['date', 'hashkey']].groupby('date').agg(['count']))

    pf = slotting(hashkey, [27, 48], 'NuSkin')


def truvision():
    pfs = [9, 48]
        
    #hashkey = generate_hashkey_ASC('data/truvision-asc-orders.csv')

    #print(hashkey.head())

    #hashkey = hashkey.set_index('data/order_number')

    #hashkey.to_csv('data/truvision_hashkey.csv')


    hashkey = load_hashkey('data/truvision_hashkey.csv')

    pf = slotting(hashkey, pfs, 'TruVision')
    #pf2 = continuous_slotting(hashkey, [48, 32], 'TruVision_continuous')

def main():
    #level_continuous()
    #waldo()
    #min_max_from_hashkey('do')
    #nuskin()
    truvision()


def test():
    hashkey = load_hashkey('data/truvision_hashkey.csv')

    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})\
        .reset_index()

    prop_65 = hashkey[hashkey['hashkey'].str.contains('PROP65')][['hashkey']]
    print(prop_65)
    print(order_count)
    

    



#main()
test()
