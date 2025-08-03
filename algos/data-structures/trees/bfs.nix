let
  tree = import ./tree.nix;
  bfs =
    queue:
    with builtins;
    if (queue == [ ]) then
      [ ]
    else
      let
        current = head queue;
        remaining_queue = tail queue;
        children = filter (x: x != null) [ current.left current.right ];
        new_queue = remaining_queue ++ children;
      in
        [ current.value ] ++ (bfs new_queue);
in
bfs [ (tree {
  value = 1;
  left = (
    tree {
      value = 2;
      left = (tree { value = 3; });
      right = (tree { value = 4; });
    }
  );
  right = (tree { value = 5; });
}) ]
