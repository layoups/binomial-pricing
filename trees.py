import numpy as np 

# presents inputs to binomial tree problem in a dictionary 
def set_inputs(asset, volatility, interest_rate, dividend, strike,  time_to_maturity, time_periods, u=None, d=None, barrier=None):
    return {"asset": asset, "sigma": volatility, "r": interest_rate, "q": dividend, "K": strike, "T": time_to_maturity, "periods": time_periods, "step": float(time_to_maturity)/time_periods, "u": u, "d": d, "barrier": barrier}

def get_A(interest_rate, dividend, volatility, step):
    return 0.5 * (np.exp(-(interest_rate - dividend)*step) + np.exp((interest_rate - dividend + volatility**2)*step))

def get_d(interest_rate, dividend, volatility, step, d):
    A = get_A(interest_rate, dividend, volatility, step)
    return  A - np.sqrt(A**2 - 1) if d == None else d

def get_u(d, u):
    return 1/d if u == None else u

# calculates CRR u
def get_crr_u(volatility, step, u):
    return np.exp(volatility*np.sqrt(step)) if u == None else u

# calculates CRR d
def get_crr_d(u, d):
    return 1/u if d == None else d

# calculates lognormal u
def get_log_u(volatility, interest_rate, step, u):
    return np.exp(interest_rate*step + volatility*np.sqrt(step)) if u == None else u

# calculates lognormal d
def get_log_d(volatility, interest_rate, step, d):
    return np.exp(interest_rate*step - volatility*np.sqrt(step)) if d == None else d

# calculates other lognormal u (with sqrt)
def get_log_u_sq(volatility, interest_rate, step, u):
    return np.exp((interest_rate - volatility**2 / 2)*step + volatility*np.sqrt(step)) if u == None else u

# calculates other lognormal d (with sqrt)
def get_log_d_sq(volatility, interest_rate, step, d):
    return np.exp((interest_rate - volatility**2 / 2)*step - volatility*np.sqrt(step)) if d == None else d

# generates binomial stock tree
def get_stock_tree(asset, u, d, time_periods):
    tree = np.zeros(2**(time_periods + 1))
    tree[1] = asset
    for t in range(2, len(tree)):
        parent = tree[t//2]
        if t % 2 == 0:
            tree[t] = parent * u
        else:
            tree[t] = parent * d
    return tree

# uses stock tree to generate p array
def get_p(interest_rate, dividend, time_periods, step, tree):
    return np.append(0, [(np.exp((interest_rate - dividend)*step) - tree[2*t + 1]/tree[t]) / (tree[2*t]/tree[t] - tree[2*t + 1]/tree[t]) for t in np.arange(1, 2**time_periods)])

# induction step for european options (call, put)
def european_option_tree(stocks, tree, p, strike, barrier, t, interest_rate, step):
    return (p[t] * tree[2*t] + (1 - p[t]) * tree[2*t + 1]) * np.exp(-interest_rate*step)

# induction step for down and out european options (call, put)
def european_down_and_out_option_tree(stocks, tree, p, strike, barrier, t, interest_rate, step):
    return european_option_tree(stocks, tree, p, strike, barrier, t, interest_rate, step) if stocks[t] > barrier else 0

# induction step for american calls
def american_call_tree(stocks, tree, p, strike, barrier, t, interest_rate, step):
    return np.max([stocks[t] - strike, (p[t] * tree[2*t] + (1 - p[t]) * tree[2*t + 1]) * np.exp(-interest_rate*step)])

# induction step for american puts 
def american_put_tree(stocks, tree, p, strike, barrier, t, interest_rate, step):
    return np.max([strike - stocks[t], (p[t] * tree[2*t] + (1 - p[t]) * tree[2*t + 1]) * np.exp(-interest_rate*step)])

#  call option at maturity
def call_at_maturity(stock, strike, barrier):
    return np.max([stock - strike, 0])

# put option at maturity
def put_at_maturity(stock, strike, barrier):
    return np.max([strike - stock, 0])

# down and out call at maturity
def down_and_out_call_at_maturity(stock, strike, barrier):
    return call_at_maturity(stock, strike, barrier) if stock > barrier else 0

# custom asset at maturity
def get_asset_at_maturity(tree, strike, barrier, time_periods, maturity_func):
    return [maturity_func(tree[t], strike, barrier) for t in np.arange(2**time_periods, 2**(time_periods + 1))]

# takes induction step function and asset t maturity function to generate the option tree
def get_asset_tree(stocks, strike, barrier, interest_rate, time_periods, p, step, maturity_func, tree_func):
    tree = np.zeros(2**(time_periods + 1))
    tree[2**time_periods:2**(time_periods + 1)] = get_asset_at_maturity(stocks, strike, barrier, time_periods, maturity_func)
    for t in np.arange(1, 2**time_periods)[::-1]:
        tree[t] = tree_func(stocks, tree, p, strike, barrier, t, interest_rate, step)
    return tree

# calculates when to exercise american option
def exercise_american(european_option, american_option):
    return np.argwhere(american_option - european_option > 0)

# generates delta hedging tree
def get_delta_tree(stocks, options, time_periods):
    return np.append(0, [(options[2*t] - options[2*t + 1]) / (stocks[2*t] - stocks[2*t + 1]) for t in np.arange(1, 2**time_periods)])

def get_gamma_tree(delta_tree, time_periods):
    return np.append(0, [(delta_tree[2*t] - delta_tree[2*t + 1]) for t in np.arange(1, 2**(time_periods - 1))])

# custom print function
def print_tree(tree, time_periods):
    for t in range(time_periods+1):
        print(tree[2**t: 2**(t+1)][::-1])

def control_variate(tree_american, tree_european, bs_european):
    return bs_european + tree_american - tree_european