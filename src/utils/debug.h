#pragma once
#include <iostream>

namespace ftxj {
    #ifndef NDEBUG
    #   define assert_msg(Expr, Msg) \
        Debug::assert_msg_(#Expr, Expr, __FILE__, __LINE__, Msg)
    #else
    #   define assert_msg(Expr, Msg) ;
    #endif

    class Debug {
    public:
        static void assert_msg_(const char* expr_str, bool expr, const char* file, int line, const char* msg) {
            if (!expr)
            {
                std::cerr << "Assert failed:\t" << msg << "\n"
                    << "Expected:\t" << expr_str << "\n"
                    << "Source:\t\t" << file << ", line " << line << "\n";
                abort();
            }
        }
    };
}