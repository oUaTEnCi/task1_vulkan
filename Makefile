task1.x: Supply.h Supply.cpp Main.cpp shaders/filters.spv
	g++ -DNDEBUG -I"3rd-party" -O3 -flto -march=native Main.cpp Supply.cpp \
		-Wl,-O1,-flto=auto -lm -lvulkan -o task1.x

shaders/filters.spv: shaders/filters.comp
	glslc -O shaders/filters.comp -o shaders/filters.spv

clean:
	rm -f task1.x gold.png gpgpu.png shaders/filters.spv