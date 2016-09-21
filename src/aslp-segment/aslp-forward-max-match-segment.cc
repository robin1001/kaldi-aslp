#include <iostream>
#include <fstream>
#include <cstdlib>
#include "forward-max-match.h"

using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cout << argv[0] << " dict text_file seg_text_file"<<std::endl;
        return 1;
    }
    Word_tree word_tree(argv[1]);
    const int MAX_BUF_SIZE = 8192;
    char text[MAX_BUF_SIZE], seg_text[MAX_BUF_SIZE*2];
    std::ifstream text_file(argv[2]);
    if (text_file.fail()) {
        perror(argv[2]);
        return 1;
    }
    std::ofstream seg_text_file(argv[3]);
    if (text_file.fail()) {
        perror(argv[3]);
        return 1;
    }
    while (text_file.getline(text, MAX_BUF_SIZE)) {
        int i = 0;
        bool has_whitespace = false;
        char* p = text;

        while (isspace(*p)) ++p;

        while (isgraph(p[i])) {
            seg_text[i] = p[i];
            ++i;
        }
		//std::cout<<seg_text<<std::endl;
        while (isspace(p[i])) {
            has_whitespace = true;
            seg_text[i] = p[i];
            ++i;
        }
        if (!has_whitespace) {
            std::cerr<<"wrong format"<<std::endl;
            std::cerr<<"format should be:\"sentenceId whitespace(e.g. \\t) content\\n\""<<std::endl;
			//std::cout<< text <<std::endl;
			//std::cout << seg_text << std::endl;
            exit(1);
        }
        word_tree.seg_word(p + i, seg_text + i);
        seg_text_file<<seg_text<<std::endl;
    }
    text_file.close();
    seg_text_file.close();
    return 0;
}
