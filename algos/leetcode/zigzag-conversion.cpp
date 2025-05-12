using Int = std::uint32_t;
using Point = std::tuple<Int, Int>;
using CharPoint = std::tuple<char, Point>;
using String = std::string;
template <typename T> using Vector = std::vector<T>;


class Solution {
public:
    string convert(string s, int numRows) {
        if (numRows == 0){
            return "";
        }
        if (numRows == 1){
            return s;
        }
        auto coordinates = Vector<CharPoint>{};
        auto coordinate = Point{0,0};
        bool isZigging = true;
        for (auto c: s){
            coordinates.push_back(CharPoint{c, coordinate});
            const auto [x,y] = coordinate;

            if (y == numRows - 1){
                isZigging = false;
                coordinate = Point{x+1, y-1};
            }else if (y == 0){
                isZigging = true;
                coordinate = Point{x,y+1};
            }else if (isZigging){
                coordinate = Point{x, y+1};
            }else {
                coordinate = Point{x+1, y-1};
            }
        }
        /* Initially used std::sort but this led to a very interesting bug on some inputs where
         * sorting occasionally resulted in out of order elements. A stable sort is required 
         * for correctness.
         */
        std::stable_sort(coordinates.begin(), 
                         coordinates.end(), 
                         [](CharPoint a, CharPoint b){ return (std::get<1>(std::get<1>(a)) < std::get<1>(std::get<1>(b))) || (std::get<0>(std::get<1>(a)) < std::get<0>(std::get<1>(b))); });

        auto result = String{};
        for (auto cp : coordinates){
            std::cout << std::get<0>(cp) << " " << std::get<0>(std::get<1>(cp)) << " " << std::get<1>(std::get<1>(cp)) << std::endl;
            result.push_back(std::get<0>(cp));
        }
        return result;
    }
