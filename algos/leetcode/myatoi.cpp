int myAtoi(string s) {
  if (s.length() == 0) {
    return 0;
  }
  auto v = std::vector<uint64_t>({});
  auto negativeFlag = false;
  auto it = s.begin();
  while (*it == 32) {
    it++;
    if (it == s.end())
      return 0;
  }
  if (*it == '-') {
    negativeFlag = true;
    it++;
  } else if (*it == '+')
    it++;

  while (it != s.end() && *it == 48) {
    it++;
  }
  while (it != s.end() && *it >= 48 && *it < 58) {
    v.push_back(static_cast<uint64_t>(*it) - 48);
    it++;
  }
  if (v.size() == 0)
    return 0;

  auto result = uint64_t(0);
  auto p = uint64_t(0);
  for (auto it = v.rbegin(); it != v.rend(); it++) {
    auto r = static_cast<uint64_t>(pow(10, p++));
    if (r > INT_MAX || (*it) * r > INT_MAX)
      return negativeFlag ? INT_MIN : INT_MAX;
    result += ((*it) * r);
  }

  return negativeFlag ? max(0 - result, static_cast<uint64_t>(INT_MIN))
                      : min(result, static_cast<uint64_t>(INT_MAX));
}
