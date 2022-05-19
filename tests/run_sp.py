from test_basic import make_kgc_connector, make_cp_connector, make_sp

if __name__ == "__main__":
    kgc_connector = make_kgc_connector()
    cp_connector = make_cp_connector()
    sp = make_sp(kgc_connector, cp_connector)
    sp.run()
