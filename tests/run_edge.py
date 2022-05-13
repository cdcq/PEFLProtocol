from main_pefl import run_edge
import sys

if __name__ == "__main__":
    print(sys.argv)
    edge_id = int(sys.argv[1])
    run_edge(edge_id=edge_id)
