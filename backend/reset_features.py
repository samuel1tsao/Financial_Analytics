import sqlite3

def reset_db():
    conn = sqlite3.connect("stock_recommender.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS company_info")
    cursor.execute("DROP TABLE IF EXISTS financial_statements")
    cursor.execute("DROP TABLE IF EXISTS corporate_actions")
    conn.commit()
    conn.close()
    print("Dropped company_info, financial_statements, and corporate_actions tables.")

if __name__ == "__main__":
    reset_db()
