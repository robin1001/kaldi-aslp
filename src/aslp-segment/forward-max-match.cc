#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <cassert>
#include "forward-max-match.h"

Hash_list::Hash_list()
:_array(NULL), _hash_size(0), _end_iter(NULL, 0, *this)
{
}

Hash_list::Hash_list(int hint)
:_end_iter(NULL, 0, *this)
{
    if (hint > 0) {
        _array = new struct Hash_node*[hint];
        memset(_array, 0, hint*sizeof(struct Hash_node*));
        _hash_size = hint;
    } else {
        _array = NULL;
        _hash_size = 0;
    }
}

Hash_list::~Hash_list()
{
    for (int i = 0; i < _hash_size; ++i)
        for (struct Hash_node* p = _array[i], *q = NULL;
            p != NULL;
            p = q)
        {
            q = p->next;
            delete p;
        }
    delete [] _array;
}

int Hash_list::resize(unsigned int new_size)
{
    if (new_size <= 0)
        return 1;
    struct Hash_node** tmp_array = new struct Hash_node*[new_size];
    memset(tmp_array, 0, new_size*sizeof(struct Hash_node*));
    //swap
    struct Hash_node** tmp = tmp_array;
    tmp_array = _array;
    _array = tmp;

    int old_hash_size = _hash_size;
    _hash_size = new_size;
    for (int i = 0; i < old_hash_size; ++i)
    {
        for (struct Hash_node* p = tmp_array[i], *q = NULL;
            p != NULL;
            p = q)
        {
            int idx = hash(p->key);
            q = p->next;
            insert(idx, p);
        }
    }
    delete [] tmp_array;
    return 0;
}

unsigned int Hash_list::hash(const char* str)
{
    const char* p = str;
    unsigned int val = 0;
    while (*p) {
        val = (val<<8) | *p++;
    }
    return val % _hash_size;
}

void Hash_list::insert(unsigned int idx, struct Hash_node* new_node)
{
    struct Hash_node* p = _array[idx];
    new_node->next = p;
    _array[idx] = new_node;
}

unsigned int Hash_list::add_elem(const char* str, Node* insert_node)
{
    unsigned int idx = hash(str);
    struct Hash_node* p = NULL;
    for (p = _array[idx];
            p != NULL;
            p = p->next)
    {
        if (!strcmp(p->key, str))
            break;
    }
    if (p == NULL) {
        struct Hash_node* pnode = new struct Hash_node;
        pnode->key = new char[strlen(str) + 1];
        strcpy(pnode->key, str);
        pnode->val = insert_node;
        pnode->next = NULL;
        insert(idx, pnode);
    } else if (p->val != insert_node) {
        delete p->val;
        p->val = insert_node;
    }
    return idx;
}

Node* Hash_list::find_elem(const char* str)
{
    unsigned int idx = hash(str);
    struct Hash_node* p = NULL;
    for (p = _array[idx]; p != NULL; p = p->next)
        if (!strcmp(p->key, str))
            break;
    if (p != NULL)
        return p->val;
    else
        return NULL;
}

inline Hash_list::iterator& Hash_list::end()
{
    return _end_iter;
}

Hash_list::iterator::iterator(Hash_list& rhs)
:_hash(rhs)
{
    _ptr = NULL;
    _idx = 0;
    for (int i = 0; i < _hash._hash_size; ++i)
    {
        if (_hash._array[i] != NULL) {
            _ptr = _hash._array[i];
            _idx = i;
        }
    }
}

Hash_list::iterator::iterator(iterator& rhs)
:_hash(rhs._hash)
{
    _ptr = rhs._ptr;
    _idx = rhs._idx;
}

Hash_list::iterator::iterator(struct Hash_node* p, unsigned int idx, Hash_list& rhs)
:_ptr(p), _idx(idx), _hash(rhs)
{
}

Hash_list::iterator& Hash_list::iterator::next()
{
    if (_ptr != NULL && _ptr->next != NULL) {
        _ptr = _ptr->next;
        return *this;
    }
    _ptr = NULL;
    for (int i = _idx + 1; i < _hash._hash_size; ++i)
        if (_hash._array[i] != NULL) {
            _ptr = _hash._array[i];
            _idx = i;
            return *this;
        }
    if (_ptr == NULL) {
        _idx = _hash._hash_size;
        return _hash._end_iter;
    }
}

bool Hash_list::iterator::operator == (const iterator& rhs)
{
    if (_ptr == rhs._ptr)
        return true;
    else
        return false;
}

Node& Hash_list::iterator::operator * ()
{
    return *_ptr->val;
}

Node::Node(int hint)
:_hash(hint)
{
    _parent = NULL;
    _cur_character = NULL;
    _is_word = false;
}

Node::Node(Node* parent, const char* character, int hint)
:_hash(hint)
{
    _parent = parent;
    _is_word = false;
    if (character == NULL)
        _cur_character = NULL;
    else {
        _cur_character = new char[strlen(character)+1];
        strcpy(_cur_character, character);
    }
}

Node* Node::add_character(const char* character)
{
    Node* pnode = new Node(this, character);
    _hash.add_elem(character, pnode);
    return pnode;
}

Node* Node::find_character(const char* character)
{
    return _hash.find_elem(character);
}

Word_tree::Word_tree(const char* dict_file)
{
    build_tree(dict_file);
}

void Node::delete_node(Node& rhs)
{
    Hash_list::iterator iter(rhs._hash);
    while (!(iter == rhs._hash.end())) {
        delete_node(*iter);
        iter.next();
    }
    delete &rhs;
}

Word_tree::~Word_tree()
{
    _root->delete_node(*_root);
}

const char* get_character(const char* text, char* character)
{
    const char* p = text;
    unsigned char ch = *p;
    int code_len = 0;
    if (!(ch & 0x80))// ascii code
        code_len = 1;
    else if (!((ch>>5)^0x06))// 2 byte utf-8 code
    {
        code_len = 2;
    }
    else if (!((ch>>4)^0x0e))
        code_len = 3;
    else if (!((ch>>3)^0x1e))
        code_len = 4;
    else
        std::cerr<<"wrong character"<<std::endl;
    *character++ = *p++;
    for (int i = 1; i < code_len; ++i)
        if (!(((unsigned char)(*p)>>6) ^ 0x02))
            *character++ = *p++;
        else
            std::cerr<<"wrong character"<<std::endl;
    *character = '\0';
    return p;
}
const int MAX_CODE_LEN = 4;
void Word_tree::add_dict_item(const char* item_text)
{
    char character[MAX_CODE_LEN + 1];//utf-8 code has [1-4] byte length 
    const char* p = item_text;
    if (_root == NULL) {
        _root = new Node(NULL, NULL);
    }
    Node* pnode = _root;
    while (*p) {
        p = get_character(p, character);
        Node* pnext_node = pnode->find_character(character);
        if (pnext_node == NULL)
            pnode = pnode->add_character(character);
        else
            pnode = pnext_node;
    }
    pnode->set_word(true);
}

Node* Word_tree::build_tree(const char* dict_file)
{
    std::ifstream dict_stream(dict_file);
    if (dict_stream.fail()) {
        perror(dict_file);
        return NULL;
    }
     
    const int MAX_BUF_SIZE = 2048;//
    char buf[MAX_BUF_SIZE];
    while (dict_stream.getline(buf, MAX_BUF_SIZE)) {
        add_dict_item(buf);
    }
}

int Word_tree::seg_word(char* text, char* seg_text)
{
    const char* p = text;
    char* pseg = seg_text;
    char character[MAX_CODE_LEN + 1];
    Node* pnode = _root;
    Node* pnext_node = NULL;
    do {
        while (*p) {
            const char* next_ch = get_character(p, character);
            int ch_size = next_ch - p;
            pnext_node = pnode->find_character(character);
            if (pnext_node != NULL) {
                pnode = pnext_node;
                strncpy(pseg, character, ch_size);
                pseg += ch_size;
                p = next_ch;
            } else if (pnode == _root) { //oov
                strncpy(pseg, character, ch_size);
                pseg += ch_size;
                *pseg++ = ' ';
                p = next_ch;
            } else if (pnode->is_word()) { //got a word 
                *pseg++ = ' ';
                pnode = _root;
            } else {
                while (pnode->parent() != _root && !pnode->is_word()) {
                    int len = strlen(pnode->cur_character());
                    pseg -= len;
                    p -= len;
                    pnode = pnode->parent();
                }
                *pseg++ = ' ';
                pnode = _root;
            }
        }

        if (pnode != _root) {
            int len = 0;
            while (pnode->parent() != _root && !pnode->is_word()) {
                len = strlen(pnode->cur_character());
                pseg -= len;
                p -= len;
                pnode = pnode->parent();
            }
            *pseg++ = ' ';
            pnode = _root;
        }
    } while (*p);

    *pseg = '\0';
}

