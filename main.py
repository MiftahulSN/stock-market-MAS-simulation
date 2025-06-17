import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class TraderAgent(Agent):
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.cash = 10000
        self.shares = 0

    def step(self):
        price = self.model.stock_price
        last_price = self.model.last_price
        initial_price = self.model.initial_price

        if self.strategy == "buy_low" and price < initial_price * 0.95:
            self.buy(price)
        elif self.strategy == "sell_high" and price > initial_price * 1.05:
            self.sell(price)
        elif self.strategy == "buy_dip" and price < last_price * 0.97:
            self.buy(price)
        elif self.strategy == "take_profit" and price > last_price * 1.03:
            self.sell(price)
        elif self.strategy == "momentum" and price > last_price:
            self.buy(price)
        elif self.strategy == "mean_reversion" and price < last_price:
            self.buy(price)
        elif self.strategy == "random":
            if self.random.random() < 0.5:
                self.buy(price)
            else:
                self.sell(price)

    def buy(self, price):
        if self.cash >= price:
            self.cash -= price
            self.shares += 1
            self.model.demand += 1
            print(f"Trader {self.unique_id} [BUY]")

    def sell(self, price):
        if self.shares > 0:
            self.cash += price
            self.shares -= 1
            self.model.supply += 1
            print(f"Trader {self.unique_id} [SELL]")


class StockMarketModel(Model):
    def __init__(self, num_traders, initial_price):
        self.num_traders = num_traders
        self.schedule = RandomActivation(self)
        self.stock_price = initial_price
        self.last_price = initial_price
        self.initial_price = initial_price
        self.demand = 0
        self.supply = 0

        strategies = [
            "buy_low", "sell_high", "random",
            "buy_dip", "take_profit", "momentum", "mean_reversion"
        ]

        for i in range(self.num_traders):
            strategy = self.random.choice(strategies)
            agent = TraderAgent(i, self, strategy)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={"Price": "stock_price"}
        )

    def step(self):
        self.demand = 0
        self.supply = 0
        self.schedule.step()

        self.last_price = self.stock_price

        if self.demand > self.supply:
            self.stock_price += 1
        elif self.supply > self.demand:
            self.stock_price -= 1

        self.stock_price = max(1, self.stock_price)
        self.datacollector.collect(self)

    def summarize_agents(self):
        print("\n\n=== Trader ===")
        for agent in self.schedule.agents:
            print(f"Trader {agent.unique_id} | Strategy: {agent.strategy} | Cash: {agent.cash:.2f} | Shares: {agent.shares}")


class StockExchange:
    def __init__(self, dataset: str, company_code: str, num_traders: int = 10):
        self.dataset = dataset
        self.company_code = company_code
        self.num_traders = num_traders
        self.stock_price = self.get_initial_price()
        self.company_shares = self.get_company_shares()
        self.init_price = self.stock_price
        self.init_shares = self.company_shares

        self.model = StockMarketModel(num_traders=num_traders, initial_price=self.stock_price)

    def get_initial_price(self):
        df = pd.read_csv(self.dataset)
        df = df[['Code', 'LastPrice']].dropna()
        selected = df[df['Code'] == self.company_code]

        if selected.empty:
            raise ValueError(f"[ERROR] company code: '{self.company_code}' not found in dataset!")

        return selected.iloc[0]['LastPrice']

    def get_company_shares(self):
        df = pd.read_csv(self.dataset)
        df = df[['Code', 'Shares']].dropna()
        selected = df[df['Code'] == self.company_code]

        if selected.empty:
            raise ValueError(f"[ERROR] company code: '{self.company_code}' not found in dataset!")

        return selected.iloc[0]['Shares']

    def get_total_agent_shares(self):
        return sum(agent.shares for agent in self.model.schedule.agents)

    def summarize_market(self):
        owned_shares = self.get_total_agent_shares()
        remaining_shares = self.company_shares - owned_shares

        print("\n=== Market Summary ===")
        print(f"Company Code     : {self.company_code}")
        print(f"Initial Price    : {self.init_price:.2f}")
        print(f"Final Price      : {self.model.stock_price:.2f}")
        print(f"Initial Shares   : {int(self.init_shares):,}")
        print(f"Owned by Traders : {int(owned_shares):,}")
        print(f"Remaining Shares : {int(remaining_shares):,}")

    def run_simulation(self, steps: int = 100):
        for _ in range(steps):
            self.model.step()

        data = self.model.datacollector.get_model_vars_dataframe()
        data.plot(title=f"Stock Price Over Time: {self.company_code}")
        plt.xlabel("Step")
        plt.ylabel("Stock Price")
        plt.grid()
        plt.tight_layout()
        plt.show()

        self.model.summarize_agents()
        self.summarize_market()


if __name__ == "__main__":
    dataset = "dataset.csv"
    stock_sim = StockExchange(dataset=dataset, company_code="PRAY", num_traders=10)
    stock_sim.run_simulation(steps=100)
