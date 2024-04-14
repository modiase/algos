class Solution {
public:

    uint32_t count_trapped(vector<vector<uint8_t>> &grid){
            uint32_t count = 0;
            const auto rows = grid.size();
            const auto cols = grid[0].size();
            for (int row = rows-1; row >= 0; row--){
                for(size_t col = 0; col < cols; col++){
                    if (grid[row][col] == 2) {
                        count++;
                    }
                }
            }
            return count;
    }
    
    void drain_grid(vector<vector<uint8_t>> &grid){
        const auto rows = grid.size();
        const auto cols = grid[0].size();
        for (size_t row = 0; row < rows; row++){
            for(size_t col = 0; col < cols; col++){
                if (grid[row][col] == 2) grid[row][col] = 0;
                else if (grid[row][col] == 1) break;
                else {
                    // grid[row][col] == 0
                    continue;
                }
            }
            for(int col = cols -1; col >= 0; col--){
                if (grid[row][col] == 2) grid[row][col] = 0;
                else if (grid[row][col] == 1) break;
                else {
                    // grid[row][col] == 0
                    continue;
                }
            }
        }
  }

    int trap(vector<int>& height) {
        const int cols = static_cast<uint16_t>(height.size());
        const int rows = *max_element(height.begin(), height.end());
        if (rows == 0) return 0;

        auto grid = vector<vector<uint8_t>>(rows, vector<uint8_t>(cols, 0));
        for (int col = 0; col < cols; col++){
            const int H = height[col];
            for (int row = 0; row < rows; row++){
                grid[row][col] = row < H ? 1 : 2;
            }
        }

        uint32_t count = 0;

        drain_grid(grid);
        count = count_trapped(grid);
        
       
        return count;
    }
};



