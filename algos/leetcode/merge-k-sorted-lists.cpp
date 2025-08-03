#include <algorithm>
#include <iostream>
#include <vector>


template <typename T> using Vector = std::vector<T>;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};


ListNode* mergeKLists(Vector<ListNode*>& lists)
{
    if (lists.empty()) return nullptr;

    auto heads = Vector<ListNode* > {};

    for (auto node : lists) {
        if (node == nullptr) continue;
        heads.push_back(node);
    }

    if (heads.empty()) return nullptr;

    const auto cmp = [](ListNode* a, ListNode* b) {
        return a->val > b->val;
    };
    std::make_heap(heads.begin(), heads.end(), cmp);

    const auto begin = heads[0];
    while (!heads.empty()) {

        std::pop_heap(heads.begin(), heads.end(), cmp);
        const auto current = heads.back();
        heads.pop_back();

        const auto nextOfCurrent = current->next;
        if (nextOfCurrent != nullptr) {
            heads.push_back(nextOfCurrent);
            std::push_heap(heads.begin(), heads.end(), cmp);
        }

        if (!heads.empty()) {
            current->next = heads[0];
        }

    }

    return begin;

}

void printLinkedList(ListNode* n)
{
    if (n == nullptr) return;

    std::cout << n->val;

    auto current = n->next;

    while (current != nullptr) {
        std::cout << "->" << current->val;
        current = current->next;
    }
    std::cout << std::endl;
}

int main()
{

    auto listOfLists = Vector<ListNode*> {};

    auto n3 = ListNode(7);
    auto n2 = ListNode(4, &n3);
    auto n1 = ListNode(1, &n2);

    auto n6 = ListNode(5);
    auto n5 = ListNode(3, &n6);
    auto n4 = ListNode(2, &n5);

    listOfLists.push_back(&n1);
    listOfLists.push_back(&n4);

    printLinkedList(&n1);
    printLinkedList(&n4);

    printLinkedList(mergeKLists(listOfLists));

    return 0;
}
