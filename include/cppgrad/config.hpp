#ifndef CPPGRAD_CONFIG_HPP
#define CPPGRAD_CONFIG_HPP

#if defined(__clang__)
	#if __has_feature(cxx_rtti)
		#define CPPGRAD_HAS_RTTI
	#endif
#elif defined(__GNUG__)
	#if defined(__GXX_RTTI)
		#define CPPGRAD_HAS_RTTI
	#endif
#elif defined(_MSC_VER)
	#if defined(_CPPRTTI)
		#define CPPGRAD_HAS_RTTI
	#endif
#endif

#endif