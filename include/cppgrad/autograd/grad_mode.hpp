#ifndef CPPGRAD_GRAD_MODE_HPP
#define CPPGRAD_GRAD_MODE_HPP

namespace cppgrad::autograd {

struct ThreadLocalGradState {
    static void set(bool new_state)
    {
        state() = new_state;
    }

    static inline bool get()
    {
        return state();
    }

private:
    static bool& state()
    {
        thread_local bool _grad_state { false };
        return _grad_state;
    }
};

struct GuardBase {
    GuardBase()
    {
        _prev_state = ThreadLocalGradState::get();
    }

    ~GuardBase()
    {
        ThreadLocalGradState::set(_prev_state);
    }

private:
    bool _prev_state;
};

struct NoGradGuard : GuardBase {
    NoGradGuard()
    {
        ThreadLocalGradState::set(false);
    }
};

struct ForceGradGuard : GuardBase {
    ForceGradGuard()
    {
        ThreadLocalGradState::set(true);
    }
};

}

#endif