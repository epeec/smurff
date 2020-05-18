#pragma once

#include <algorithm>
#include <numeric>
#include <thread>
#include <map>
#include <cassert>

template <typename T>
class thread_vector
{
public:
    thread_vector(const T &t = T())
    {
        init(t);
    }
    template <typename F>
    T combine(F f) const
    {
        T sum = _i;
        for(const auto &v : _m) sum = f(sum, v.second);
        return sum;
    }

    T combine() const
    {
        return combine([](const T &a, const T &b) { return a+b; });
    }

    T &local()
    {
        auto pos = _m.find(std::this_thread::get_id());

        if (pos == _m.end())
        {
            auto r = _m.insert(std::make_pair(std::this_thread::get_id(), _i));
            pos = r.first;
        }

        return pos->second;
    }
    void reset()
    {
        _m.clear();
    }

    template <typename F>
    T combine_and_reset(F f) const
    {
        T ret = combine(f);
        reset();
        return ret;
    }

    T combine_and_reset()
    {
        T ret = combine();
        reset();
        return ret;
    }

    void init(const T &t = T())
    {
        _i = t;
        reset();
    }

    typedef typename std::map<std::thread::id, T> container_type;
    typedef typename container_type::const_iterator const_iterator;

    const_iterator begin() const
    {
        return _m.begin();
    }

    const_iterator end() const
    {
        return _m.end();
    }

    typename std::vector<T>::size_type size() const 
    {
        return _m.size();
    }

private:
    std::map<std::thread::id, T> _m;
    T _i;
};