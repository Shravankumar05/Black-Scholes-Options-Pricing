import sqlite3

def create_table(db_path="data.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS Calculations (
        calc_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_price       REAL,
        strike_price      REAL,
        interest_rate     REAL,
        volatility        REAL,
        time_to_expiry    REAL,
        call_buy_price    REAL,
        put_buy_price     REAL,
        timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    """)
    c.execute("""
      CREATE TABLE IF NOT EXISTS Outputs (
        output_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        calc_id           INTEGER,
        sigma_shock       REAL,
        spot_shock        REAL,
        option_price      REAL,
        is_call           BOOLEAN,
        FOREIGN KEY(calc_id) REFERENCES Calculations(calc_id)
      )
    """)
    conn.commit()
    conn.close()


def insert_calculation(params: dict, db_path="data.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
            INSERT INTO Calculations (
                stock_price, strike_price, interest_rate, volatility, time_to_expiry, call_buy_price, put_buy_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
              """, (
        params['stock_price'], params['strike_price'], params['interest_rate'], params['volatility'], params['time_to_expiry'], params['call_buy_price'], params['put_buy_price']
    ))
    conn.commit()
    calc_id = c.lastrowid
    conn.close()
    return calc_id

def insert_output(calc_id: int, sigma_shock: float, spot_shock: float, option_price: float, is_call: bool, db_path="data.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
      INSERT INTO Outputs
        (calc_id, sigma_shock, spot_shock, option_price, is_call)
      VALUES (?, ?, ?, ?, ?)
    """, (calc_id, sigma_shock, spot_shock, option_price, int(is_call)))
    conn.commit()
    conn.close()

def insert_outputs_bulk(calc_id: int, data_tuples: list, db_path="data.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    bulk_data = [(calc_id, sigma, spot, price, int(is_call)) for sigma, spot, price, is_call in data_tuples]
    c.executemany("""
      INSERT INTO Outputs
        (calc_id, sigma_shock, spot_shock, option_price, is_call)
      VALUES (?, ?, ?, ?, ?)
    """, bulk_data)
    
    conn.commit()
    conn.close()
