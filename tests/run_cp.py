from test_basic import make_kgc_connector, make_cp

if __name__ == "__main__":
    kgc_connector = make_kgc_connector()
    cp = make_cp(kgc_connector)
    cp.run()
