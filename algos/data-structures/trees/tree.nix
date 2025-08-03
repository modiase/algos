let
  tree =
    {
      value,
      left ? null,
      right ? null,
    }:
    {
      value = value;
      left = left;
      right = right;
    };
in
tree
