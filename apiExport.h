#pragma once

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#define API_EXPORT __declspec(dllexport)
#define API_IMPORT __declspec(dllimport)
#define API_LOCAL
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define API_EXPORT __attribute__((visibility("default")))
#define API_IMPORT __attribute__((visibility("default")))
#define API_LOCAL __attribute__((visibility("hidden")))
#else
#define API_EXPORT
#define API_IMPORT
#define API_LOCAL
#endif

#ifdef __cplusplus
#define API_DEMANGLED extern "C"
#else
#define API_DEMANGLED
#endif

#ifdef API_EXPORTS
#define API_PUBLIC API_EXPORT
#else
#define API_PUBLIC API_IMPORT
#endif