
from evonet.trader.data_loader import DataFetcher

def check_aapl():
    print("Fetching AAPL...")
    f = DataFetcher("AAPL", provider="yf")
    df = f.fetch_data()
    print(f"AAPL Rows: {len(df) if df is not None else 'None'}")
    
    if df is not None:
        print(df.tail())

if __name__ == "__main__":
    check_aapl()
