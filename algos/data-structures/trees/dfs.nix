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

  dfs =
    tree:
    with builtins;
    if (tree == null) then
      [ ]
    else
      (concatLists (
        map (child: dfs child) (
          filter (x: x != null) [
            tree.left
            tree.right
          ]
        )
      ))
      ++ [ tree.value ];
in
dfs (tree {
  value = 1;
  left = (
    tree {
      value = 2;
      left = (
        tree {
          value = 3;
        }
      );
      right = (
        tree {
          value = 4;
        }
      );
    }
  );
  right = (
    tree {
      value = 5;
    }
  );
})
