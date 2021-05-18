import argparse

def main(args):
  #print(s)
  print(args.verbose)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Citrus Disease Classification Training Pipeline")

  parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
  main(parser.parse_args())