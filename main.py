from alphas import get_alpha


def load_data():
    """

    :return:
    """
    idx_path = '/shared/JY_Data/eod/'
    dates = np.load(idx_path + 'dates.npy').astype(str)

    ticker_names = np.load(idx_path + 'ticker_names.npy').astype(str)
    openPrice = pd.DataFrame(np.load(idx_path + 'OpenPrice.npy'), index=ticker_names, columns=dates)
    closePrice = pd.DataFrame(np.load(idx_path + 'ClosePrice.npy'), index=ticker_names, columns=dates)
    highPrice = pd.DataFrame(np.load(idx_path + 'HighestPrice.npy'), index=ticker_names, columns=dates)
    lowPrice = pd.DataFrame(np.load(idx_path + 'LowestPrice.npy'), index=ticker_names, columns=dates)
    volume = pd.DataFrame(np.load(idx_path + 'Volume.npy'), index=ticker_names, columns=dates)
    # turnover = pd.DataFrame(np.load(idx_path + 'Turnover.npy'), index=ticker_names, columns=dates)
    vwap = pd.DataFrame(np.load(idx_path + 'VWAP.npy'), index=ticker_names, columns=dates)
    cap = pd.DataFrame(np.load('/shared/JY_Data/fundmental/CAPQ0_MKTCAP.npy'), index=ticker_names, columns=dates)

    df = pd.DataFrame(index=ticker_names, columns=dates).fillna(1)
    df = df.stack().to_frame().reset_index().drop([0], axis=1)
    df.columns = ['Code', 'Date']
    names = ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'Volume', 'VWAP', 'MktCap']
    for df_, name in zip([openPrice, closePrice, highPrice, lowPrice, volume, vwap, cap], names):
        df_ = df_.stack().to_frame()
        df_.reset_index(inplace=True)
        df_.columns = ['Code', 'Date', name]
        df = pd.merge(df, df_, how='left', on=['Code', 'Date'])

    return df.set_index(['Code', 'Date'])


if __name__ == "__main__":
    data = load_data()
    data = data.sort_index()
    factor = get_alpha(data)
