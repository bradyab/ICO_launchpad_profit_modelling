import pandas as pd
import requests
import json
import os
import itertools

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
import ipdb
import re

# runtime-saving modularization
fetch_new_data = 0
clean_data = 0
calculate_days_since = 0

if fetch_new_data:
    # ico_data = requests.get(
    #     'https://api.cryptorank.io/v1/currencies/crowdsales/',
    #     params={
    #         'api_key':
    #         'c1aff9e4dcf5c8f56137f884397c38bbe11583d356b8adaad04bbda375ed',
    #         "state": "all",
    #         "limit": "10000000"
    #     })
    # ico_data = json.loads(ico_data.content)['data']
    # # convert data into pandas dataframe
    # df = pd.DataFrame.from_dict(ico_data)
    # df.to_csv("ico_data_may18.csv")
    df = pd.read_csv("ico_data_may18.csv")
    # filter df to unique 'id'
    df_unique = df.drop_duplicates(subset=['id'], keep='first')
    # get these duplicates? duplicate rounds?
    ico_ids = df_unique['id'].tolist()
    import csv

    f = open('platform_coins.csv')
    l = []
    f = csv.reader(f)
    for i in f:
        l.append(int(i[0]))

    combined_ids = ico_ids + l
    df_responses = pd.DataFrame()

    # for id in combined_ids:
    #     response = requests.get(
    #         'https://api.cryptorank.io/v1/currencies/{}/sparkline'.format(id),
    #         params={
    #             "api_key":
    #             "c1aff9e4dcf5c8f56137f884397c38bbe11583d356b8adaad04bbda375ed",
    #             "to": datetime.today(),
    #             "from": datetime.today() - timedelta(days=89),
    #             "interval": "1d"
    #         })
    #     # change interval to get precise vesting time price???
    #     jsoned = pd.DataFrame(response.json())
    #     jsoned = jsoned.transpose()[["dates", "volumes", "prices"]].drop(index='status')
    #     # KeyError: "None of [Index(['dates', 'volumes', 'prices'], dtype='object')] are in the [columns]" probably means the response was bad
    #     jsoned['id'] = id
    #     print(id)
    #     df_responses = df_responses.append(jsoned, ignore_index=True)
    
    # converters bc saving to CSV adds quotes around objects
    df_responses = pd.read_csv("df_responses_may18.csv", converters={'dates': pd.eval, 'prices': pd.eval, 'volumes': pd.eval})
    # incorporate ico data from spreadsheets that arent found in api?

    s1 = pd.DataFrame(df_responses.pop('prices').values.tolist(), index=df_responses.index).stack().rename('prices').reset_index(level=1, drop=True)
    s2 = pd.DataFrame(df_responses.pop('dates').values.tolist(), index=df_responses.index).stack().rename('dates').reset_index(level=1, drop=True)
    s3 = pd.DataFrame(df_responses.pop('volumes').values.tolist(), index=df_responses.index).stack().rename('volumes').reset_index(level=1, drop=True)
    new_df = df_responses.join(pd.concat([s1, s2, s3], axis=1))
    
    # for index, cols in df_responses.iterrows():
    #     extract_df = pd.DataFrame({'dates': cols['dates'], 'volumes': cols['volumes'], 'prices': cols['prices'], 'id': cols['id']}, index=[0])
    #     extract_df = pd.concat(
    #         [
    #             extract_df,
    #             cols.drop(['dates', 'volumes', 'prices', 'id']).to_frame().T
    #         ],
    #         axis=1).fillna(method='ffill').fillna(method='bfill')
    #     new_df = pd.concat([new_df, extract_df], ignore_index=True)

    new_df['id'] = new_df['id'].astype(np.int64)
    new_df = new_df.drop(columns=[0])
    new_df.to_csv("historical_data_exp.csv")

if clean_data:

    prices_df = pd.read_csv('historical_data.csv').drop(columns=['Unnamed: 0'])
    new_prices_df = pd.read_csv('historical_data_exp.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    prices_df['dates'] = pd.to_datetime(prices_df['dates'])
    new_prices_df['dates'] = pd.to_datetime(new_prices_df['dates'])
    merged_prices_df = pd.merge(prices_df, new_prices_df, how='outer', on=['id','dates', 'volumes', 'prices'])
    
    # ensure merge wasn't messy
    # nona = merged_prices_df.dropna(subset=['prices_x', 'prices_y'])[['id', 'dates','prices_x', 'prices_y']]
    # diff = nona.loc[~(nona['prices_x'] == nona['prices_y'])] 
    
    prices_df = merged_prices_df.sort_values(['id', 'dates'], ascending=[True, True])
    ico_data_df = pd.read_csv("ico_data_may18.csv")
    mapping_df = pd.read_csv("launchpad_ico_mapping.csv")
    # ipdb.set_trace()
    for index, row in mapping_df.iterrows():
        if row['symbol'] is np.nan:
            mymatch = re.compile('([A-Z]{2,})')
            symbol = mymatch.findall(row['Name'])
    
            try:
                mapping_df.loc[index, 'symbol'] = symbol[-1]
            except:
                mapping_df.loc[index, 'symbol'] = ''
            # if len(symbol) >1:
            #     ipdb.set_trace()
            # account for numbers???

    # explode values
    ico_data_df['rounds'] = ico_data_df['rounds'].str.replace("'", '"')
    # str to dict
    ico_data_df['rounds'] = ico_data_df['rounds'].apply(json.loads)
    # expand the dict in col rounds into new rows, copying all other columns
    explode_df = ico_data_df.explode('rounds')
    explode_df = explode_df.dropna(subset=['rounds'])
    # this order is different for old ico data csv
    explode_df[['endDate', 'type', 'tokensForSale', 'values','startDate', 'ieoPlatformExchange']] = pd.DataFrame(explode_df.rounds.tolist(), index=explode_df.index)
    # select most recent round for each coin that ICOd. conservative
    sorted_df = explode_df.sort_values(['id', 'endDate'],
                                       ascending=[True, False])
    dedup_df = sorted_df.drop_duplicates(subset='id', keep='first')
    # need to filter out coins where sales ended before we have historical price data ??? and those with missing price data
    # dedup_df[(dedup_df['endDate'] >= '2019-12-06 00:00:00+0000')]
    # filter dataframe so that column 'category' contains "ICO", "IDO", or "IEO"
    # all the other types are private rounds
    # dynamically set columns in future, as API output changes
    USD_df = dedup_df[dedup_df['type'].isin(['ICO', 'IEO', 'IDO'])]
    USD_df['USD'] = USD_df['values'].apply(pd.Series)['USD']
    USD_df[['ICO_price', 'intended_raise','hardCap', 'softCap', 'actual_raised']] = USD_df['USD'].apply(pd.Series).drop(columns=[0])

    ROI_df = USD_df.drop(columns=["Unnamed: 0", 'slug', 'rounds', 'USD', 'values'])
    prices_df = prices_df.set_index('id')
    ROI_df['platform'] = ""
    ROI_df['platform'] = ROI_df['platform'].astype('object')
    i=0
    for index, row in ROI_df.iterrows():
        # map launchpad to ICO
        # we don't have prices for all mapping df coins
        # set id to index?
        map = mapping_df.loc[mapping_df['symbol'].str.fullmatch(
            row['symbol'], case=True) & mapping_df['Name'].str.contains(
            row['name'])]
        pricem = prices_df[prices_df.index.isin([row['id']])]
        ico = ROI_df.loc[ROI_df['symbol'].str.fullmatch(
            row['symbol'], case=True)]
        if map.shape[0] > 0 and row['symbol'] not in ['REAL', 'HERO'] and not pricem.empty:
            ROI_df.at[index, 'platform'] = map['platform'].tolist()
            # print(i)
            i=i+1

        # first sell
        try:
            # first price date may be limited by data lookback period. filter out these cases later
            ROI_df.loc[index, 'date_1'] = prices_df.loc[row['id']].iloc[0]['dates']
            ROI_df.loc[index, 'price_1'] = prices_df.loc[
                row['id']].iloc[0]['prices'].item()
            ROI_df.loc[index, 'sale_1'] = 0.186875 * ROI_df.loc[index, 'price_1']
            ROI_df.loc[index, 'cumulative_realized_ROI_1'] = ROI_df.loc[
                index, 'sale_1'] / row['ICO_price']
        except:
            ROI_df.loc[index, 'date_1'] = np.nan
            ROI_df.loc[index, 'price_1'] = np.nan
            ROI_df.loc[index, 'sale_1'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_1'] = np.nan

        # second sell
        # np.nans are often points that were in the future at the time of data snapshot
        try:
            ROI_df.loc[index, 'date_2'] = ROI_df.loc[index, 'date_1'] + timedelta(days=30)
            ROI_df.loc[index, 'price_2'] = prices_df.loc[
                (prices_df.index == row["id"])
                & (prices_df['dates'] == ROI_df.loc[index, 'date_2']
                   )]['prices'].item()
            ROI_df.loc[index, 'sale_2'] = 0.19515625 * ROI_df.loc[index, 'price_2']
            ROI_df.loc[index, 'cumulative_realized_ROI_2'] = (
                ROI_df.loc[index, 'sale_2'] +
                ROI_df.loc[index, 'sale_1']) / row['ICO_price']
    
            ROI_df.loc[index,
                       'date_3'] = ROI_df.loc[index, 'date_2'] + timedelta(days=30)
        except:
            ROI_df.loc[index, 'date_2'] = np.nan
            ROI_df.loc[index, 'price_2'] = np.nan
            ROI_df.loc[index, 'sale_2'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_2'] = np.nan

        # third sell                
        try:
            ROI_df.loc[index, 'date_3'] = ROI_df.loc[index, 'date_2'] + timedelta(days=30)
            ROI_df.loc[index, 'price_3'] = prices_df.loc[
            (prices_df.index == row["id"])
            & (prices_df['dates'] == ROI_df.loc[index, 'date_3']
               )]['prices'].item()
            ROI_df.loc[index,
                       'sale_3'] = 0.1641444444 * ROI_df.loc[index, 'price_3']
            ROI_df.loc[index, 'cumulative_realized_ROI_3'] = (
                ROI_df.loc[index, 'sale_3'] + ROI_df.loc[index, 'sale_2'] +
                ROI_df.loc[index, 'sale_1']) / row['ICO_price']
    
        except:
            ROI_df.loc[index, 'date_3'] = np.nan
            ROI_df.loc[index, 'price_3'] = np.nan
            ROI_df.loc[index, 'sale_3'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_3'] = np.nan

        # fourth sell 
        try:
            ROI_df.loc[index, 'date_4'] = ROI_df.loc[index, 'date_3'] + timedelta(days=30)
            ROI_df.loc[index, 'price_4'] = prices_df.loc[
            (prices_df.index == row["id"])
            & (prices_df['dates'] == ROI_df.loc[index, 'date_4']
               )]['prices'].item()
            ROI_df.loc[index,
               'sale_4'] = 0.1612261905 * ROI_df.loc[index, 'price_4']
            ROI_df.loc[index, 'cumulative_realized_ROI_4'] = (
                ROI_df.loc[index, 'sale_4'] + ROI_df.loc[index, 'sale_3'] +
                ROI_df.loc[index, 'sale_2'] +
                ROI_df.loc[index, 'sale_1']) / row['ICO_price']
        # date might not have happened yet, so price will be missing
        except:
            ROI_df.loc[index, 'date_4'] = np.nan
            ROI_df.loc[index, 'price_4'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_4'] = np.nan
            ROI_df.loc[index, 'sale_4'] = np.nan

        # fifth sell
        try:
            ROI_df.loc[index, 'date_5'] = ROI_df.loc[index, 'date_4'] + timedelta(days=30)
            ROI_df.loc[index, 'price_5'] = prices_df.loc[
                (prices_df.index == row["id"])
                & (prices_df['dates'] == ROI_df.loc[index, 'date_5']
                   )]['prices'].item()
            
            ROI_df.loc[index,
                       'sale_5'] = 0.1605972222 * ROI_df.loc[index, 'price_5']
            ROI_df.loc[index, 'cumulative_realized_ROI_5'] = (
                ROI_df.loc[index, 'sale_5'] + ROI_df.loc[index, 'sale_4'] +
                ROI_df.loc[index, 'sale_3'] + ROI_df.loc[index, 'sale_2'] +
                ROI_df.loc[index, 'sale_1']) / row['ICO_price']
        except:
            ROI_df.loc[index, 'date_5'] = np.nan
            ROI_df.loc[index, 'price_5'] = np.nan
            ROI_df.loc[index, 'sale_5'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_5'] = np.nan

        # sixth sell
        try:
            ROI_df.loc[index, 'date_6'] = ROI_df.loc[index, 'date_5'] + timedelta(days=30)
            ROI_df.loc[index, 'price_6'] = prices_df.loc[
                (prices_df.index == row["id"])
                & (prices_df['dates'] == ROI_df.loc[index, 'date_6']
                   )]['prices'].item()
            ROI_df.loc[index,
                       'sale_6'] = 0.1320380952 * ROI_df.loc[index, 'price_6']
            ROI_df.loc[index, 'cumulative_realized_ROI_6'] = (
                ROI_df.loc[index, 'sale_6'] + ROI_df.loc[index, 'sale_5'] +
                ROI_df.loc[index, 'sale_4'] + ROI_df.loc[index, 'sale_3'] +
                ROI_df.loc[index, 'sale_2'] +
                ROI_df.loc[index, 'sale_1']) / row['ICO_price']
        except:
            ROI_df.loc[index, 'date_6'] = np.nan
            ROI_df.loc[index, 'price_6'] = np.nan
            ROI_df.loc[index,'sale_6'] = np.nan
            ROI_df.loc[index, 'cumulative_realized_ROI_6'] = np.nan

    ROI_df.to_csv("ROI_df.csv")
    prices_df.to_csv("prices_df_augmented.csv")

    ###########
if calculate_days_since:
    prices_df = pd.read_csv("prices_df.csv")
    ROI_df = pd.read_csv("ROI_df.csv")
    prices_df['dates'] = pd.to_datetime(prices_df['dates'], utc=True)
    ROI_df['endDate'] = pd.to_datetime(ROI_df['endDate'], utc=True)
    prices_df = prices_df[prices_df['prices'].notna()]
    prices_df = prices_df[prices_df['id'].isin(ROI_df.id)]
    
    for index, row in prices_df.iterrows():
        prices_df.loc[index, 'days_since_first_price'] = row['dates'] - prices_df.loc[prices_df.id == row['id']]['dates'].min()
        # use this to select actual value of timedif, not the series
        helper = row['dates'] - ROI_df.loc[ROI_df.id == row['id']].endDate
        try:
            prices_df.at[index, 'days_since_ICO_end'] = int(helper[0].split(' ')[0])
        except:
            prices_df.at[index, 'days_since_ICO_end'] = np.nan
        try:
            prices_df.loc[index, 'symbol'] = ROI_df.loc[ROI_df.id == row['id']].symbol.head(1)[0]
        except:
            prices_df.loc[index, 'symbol'] = np.nan

############
            
prices_df = pd.read_csv("prices_df_augmented.csv")
ROI_df = pd.read_csv("ROI_df.csv")
prices_df['dates'] = pd.to_datetime(prices_df['dates'], utc=True)
ROI_df['endDate'] = pd.to_datetime(ROI_df['endDate'], utc=True)
# filter out coins where we don't have an accurate first price - where we have a price on our furthest data availability date
ROI_df = ROI_df[~ROI_df['id'].isin(ROI_df[ROI_df['date_1'] < '2022-02-19 00:01:00+0000'].id.unique())]

merged_df = pd.merge(prices_df, ROI_df, how='outer', on=['id'])
merged_df['price_divided_by_ICO_price'] = merged_df['prices'] / merged_df['ICO_price']
merged_df['dates'] = merged_df['dates'].dt.date
merged_df['days_since_ICO_end'] = merged_df['days_since_ICO_end'].str.split(' ')
merged_df[['days_since_ICO_end','trash']] = pd.DataFrame(merged_df.days_since_ICO_end.fillna('').tolist(), index=merged_df.index)
merged_df['days_since_first_price'] = merged_df['days_since_first_price'].str.split(' ')
merged_df[['days_since_first_price','trash2']] = pd.DataFrame(merged_df.days_since_first_price.fillna('').tolist(), index=merged_df.index)

# create df for platform level data
platforms_df = pd.DataFrame()
platforms_df['platform'] = ""
platforms=list(map(lambda x: x.strip('][').split(', '), np.delete(ROI_df.platform.unique(), 0)))
joined_platforms = list(itertools.chain.from_iterable(platforms))

# calculate average and median ROI per sale per platform
for n in range(1,7):
    for platform in joined_platforms:
        platform = platform.replace("'","")
        platforms_df.loc[platform, 'sale_'+str(n)+'_average_cumulative_ROI'] = ROI_df[ROI_df.platform.str.contains(platform, na=False)]['cumulative_realized_ROI_'+str(n)].mean()
        platforms_df.loc[platform, 'sale_'+str(n)+'_median_cumulative_ROI'] = ROI_df[ROI_df.platform.str.contains(platform, na=False)]['cumulative_realized_ROI_'+str(n)].median()
        platforms_df.loc[platform, 'sale_'+str(n)+'_count'] = ROI_df[ROI_df.platform.str.contains(platform, na=False)]['cumulative_realized_ROI_'+str(n)].count()

# get data into plottable format
old_avgs_stacked = pd.DataFrame(platforms_df[['sale_1_average_cumulative_ROI', 'sale_2_average_cumulative_ROI', 'sale_3_average_cumulative_ROI', 'sale_4_average_cumulative_ROI', 'sale_5_average_cumulative_ROI', 'sale_6_average_cumulative_ROI']].stack())
avgs_stacked = old_avgs_stacked.reset_index()
avgs_stacked['sale'] = avgs_stacked['level_1'].str.extract('(\d+)', expand=False).reset_index().drop(columns=['index']).astype(int)
medians_stacked = pd.DataFrame(platforms_df[['sale_1_median_cumulative_ROI', 'sale_2_median_cumulative_ROI', 'sale_3_median_cumulative_ROI', 'sale_4_median_cumulative_ROI', 'sale_5_median_cumulative_ROI', 'sale_6_median_cumulative_ROI']].stack())
old_platforms_stacked = pd.merge(avgs_stacked, medians_stacked.reset_index(), left_index=True, right_index=True).rename(columns={'level_0_x': 'symbol', "0_x": "average_cumulative_realized_ROI", "0_y": "median_cumulative_realized_ROI"})
count_stacked = pd.DataFrame(platforms_df[['sale_1_count', 'sale_2_count', 'sale_3_count', 'sale_4_count', 'sale_5_count', 'sale_6_count']].stack())
# filter out small sample sizes
platforms_stacked = pd.merge(old_platforms_stacked, count_stacked[count_stacked[0]>2].reset_index(), left_index=True, right_index=True)

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(2,2)

gs.update(left=0.1,right=0.9,top=0.965,bottom=0.03,wspace=0.8,hspace=1)
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0,0], hspace=0)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0,1], hspace=0)
gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1,1], hspace=0)
gs4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1,0], hspace=0)
pl.figure()


###### all platforms avg cumulative_ROI over unlock period
# ipdb.set_trace()
platforms_stacked_grouped = platforms_stacked.groupby('symbol')
ax = pl.subplot(gs1[0,0])
for key, group in platforms_stacked_grouped:
    group.plot('sale', 'average_cumulative_realized_ROI', label=key, ax=ax)

ax.set_yscale('log')
plt.xticks(rotation=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('avg cumulative ROI per platform')
plt.xlabel("""Unlock #
           
Unlocks are generally every month after the TGE.

below are the best and worst platforms based on final ROI:           
  
platform   average
   GZONE   1.717616   
    BSCS   1.356817   
    PAID   1.011838   
     ...        ...   
  pancake   0.107957   
    duck   0.096859   
  zendit   0.008134   
""")
plt.ylabel("avg cumulative ROI")

######

# get data into plottable format
ROI_stacked = pd.merge(pd.DataFrame(ROI_df[['cumulative_realized_ROI_1', 'cumulative_realized_ROI_2', 'cumulative_realized_ROI_3', 'cumulative_realized_ROI_4', 'cumulative_realized_ROI_5', 'cumulative_realized_ROI_6']].stack().droplevel(level=1)), pd.DataFrame(ROI_df[['date_1', 'date_2', 'date_3', 'date_4', 'date_5', 'date_6']].stack().droplevel(level=1)), left_index=True, right_index=True).rename(columns={"0_x": "cumulative_ROI", "0_y": "date"})
ROI_stacked = pd.merge(ROI_df[['id', 'symbol']], ROI_stacked, left_index=True, right_index=True)

###### all coins cumulative_ROI over abs time
ROI_stacked_grouped = ROI_stacked.head(100).groupby(['symbol'])
ax = pl.subplot(gs2[0,0]) 
for key, group in ROI_stacked_grouped:
    group.plot('date', 'cumulative_ROI', label=key, ax=ax)

ax.set_yscale('log')
plt.xticks(rotation=30)
plt.title('cumulative ROI over time, all coins')
plt.xlabel("""Date""")
plt.ylabel("cumulative ROI")
ax.get_legend().remove()
#######

# this is useless and misleading till we have more data
###### price_divided_by_ICO_price over time since first price
merged_df['days_since_ICO_end'] = merged_df['days_since_ICO_end'].apply(pd.to_numeric)
merged_df['days_since_first_price'] = merged_df['days_since_first_price'].apply(pd.to_numeric)
grouped = merged_df.dropna(subset=['price_divided_by_ICO_price','days_since_first_price']).groupby(['symbol_y'])
# ipdb.set_trace()
ax = pl.subplot(gs3[0,0]) 
i=0
for key, group in grouped:
    # ipdb.set_trace()
    i = i+1
    # gets hung on last group
    if i==150:
        break
    group.plot('days_since_first_price', 'price_divided_by_ICO_price', label=key, ax=ax, alpha=0.3)
    
ax.set_yscale('log')
plt.xticks(rotation=30)
plt.title('price / ICO price vs time since ICO')
plt.xlabel("""""")
plt.ylabel("price / ICO price")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

###### display images from spreadsheet

import matplotlib.image as mpimg
img = mpimg.imread('per_tier.png')
plt.figure(0)
plt.imshow(img)

img2 = mpimg.imread('AVG_ATH_ROI_per_launchpad.png')
plt.figure()
plt.imshow(img2)

plt.show()

########
# next steps

# hard
# add tiers - get price of stake token on start date, add stake cost to denom of ROI, add allocation to num of ROI
# get more data. 
# add stake appreciation to ROI

# medium
# btc-normalized price over absolute time
# average ROI per IGO date vs btc
# create PNL equivaluent charts
# look at IEOs

# easy
# normalized price over relative time "do prices ususally tank after ICO"
# per-pad stats: ATH/avg/realized ROI for each, how that correlates to number of ICOs, 
