#ifndef ASLP_SEGMENT_FORWARD_MAX_MATCH_H_
#define ASLP_SEGMENT_FORWARD_MAX_MATCH_H_

class Node;

class Hash_list {
public:
    struct Hash_node {
        char* key;
        Node* val;
        Hash_node* next;
    };
    Hash_list();
    Hash_list(int hint);
    ~Hash_list();
    unsigned int add_elem(const char* str, Node* insert_node);
    Node* find_elem(const char* str);
    class iterator;
    friend class iterator;
    class iterator {
    public:
        iterator(Hash_list& rhs);
        iterator(iterator& itr);
        iterator(struct Hash_node* p, unsigned int idx, Hash_list& hash);
        iterator& next();
        bool operator == (const iterator& rhs);
        Node& operator * ();
    private:
        struct Hash_node* _ptr;
        unsigned int _idx;
        Hash_list& _hash;
    };
    iterator& end();
private:
    unsigned int hash(const char* str);
    int resize(unsigned int new_size);
    void insert(unsigned int idx, struct Hash_node* new_node);
    struct Hash_node** _array;
    unsigned int _hash_size;
    //for iterator
    iterator _end_iter;
};

class Node {
public:
    Node(int hint=1024);
    Node(Node* parent, const char* character, int hint=1024);
    ~Node() { delete [] _cur_character; }
    Node* add_character(const char* character);
    Node* find_character(const char* character);
    void delete_node(Node& rhs);
    bool is_word() { return _is_word; }
    void set_word(bool flag) { _is_word = flag; }
    Node* parent() { return _parent; }
    const char* cur_character() { return _cur_character; }
private:
    Node* _parent;
    char* _cur_character;
    Hash_list _hash;
    bool _is_word;
};

class Word_tree {
public:
    Word_tree(const char* dict_file);
    ~Word_tree();
    int seg_word(char* text, char* seg_text);
private:
    void add_dict_item(const char* item_text);
    Node* build_tree(const char* dict_file);
    Node* _root;
};

#endif
