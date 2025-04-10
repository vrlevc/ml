#include <iostream>
#include <vector>
#include <string>
using namespace std;

int minDistance(string word1, string word2) {
    int m = word1.size();
    int n = word2.size();

    // Create a DP table
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // Initialize the base cases
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i; // Deleting all characters from word1
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j; // Inserting all characters into word1
    }

    // Fill the DP table
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1]; // Characters match, no operation needed
            } else {
                dp[i][j] = 1 + min({dp[i - 1][j],    // Delete
                                    dp[i][j - 1],    // Insert
                                    dp[i - 1][j - 1] // Replace
                                   });
            }
        }
    }

    return dp[m][n];
}

int main() {
    string word1, word2;
    cout << "Enter word1: ";
    cin >> word1;
    cout << "Enter word2: ";
    cin >> word2;

    int result = minDistance(word1, word2);
    cout << "The minimum number of operations required: " << result << endl;

    return 0;
}