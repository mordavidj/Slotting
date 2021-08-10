import pandas as pd
import matplotlib.pyplot as plt
from Pickface import *



def evaluate(pfs, hashkey:pd.DataFrame, lol:bool = False):
    '''Evaluate a pickface using a hashkey of orders.

    '''
    print('\nEvaluating pickfaces')
    ord_sum = len(hashkey)

    if type(pfs) is not list:
        pfs = [pfs]
    #hashkey = remove_lol(hashkey)

    print('\nTotal Orders: {0:,}'.format(ord_sum))
    by_date = hashkey['date'].value_counts().reset_index()\
        .rename(columns = {'date': 'count', 'index': 'date'})\
        .sort_values('date').set_index('date')

    by_date.plot(kind = 'hist', legend = None)
    by_date.plot(kind = 'bar', legend = None)
    plt.show()
    if lol:
        hashkey = remove_lol(hashkey)

    by_date = hashkey['date'].value_counts().reset_index()\
        .rename(columns = {'date': 'count', 'index': 'date'})\
        .sort_values('date').set_index('date')

    by_date.plot(kind = 'hist', legend = None)
    by_date.plot(kind = 'bar', legend = None)
    plt.show()
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    for pf in pfs:
        items = pd.DataFrame(pf.list_items(), columns = ['item_id'])\
            .set_index('item_id')
        items['orders'] = 0
        #print(items)

        
        for index, row in order_count.iterrows():
            if all(x in items.index.tolist() for x in index.split(';')):
                order_count.at[index, 'visited'] = True
                for o in index.split(';'):
                    items.at[o, 'orders'] += row['order_count']

                
                #print(index)
        parts = [[] for p in range(pf.num_bays)]
        for p in len(parts):
            pass

        print(items)
        ord_serv = items.orders.sum()

        ord_per = ord_serv / ord_sum
        print('\nOrders Served by PF: {0:,} ({1:.2%})'.format(ord_serv, 
                                                              ord_per))

        visited = list(order_count[order_count.visited == True].index)
        if visited:
            sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
            sub_val_count = sub_hashkey['date'].value_counts()

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

        parts
        order_count = order_count[~order_count.visited]

    rem_sum = order_count.order_count.sum()
    rem_per = rem_sum / ord_sum
    print('\nOrders Remaining: {0:,} ({1:.2%})'.format(rem_sum, rem_per))
