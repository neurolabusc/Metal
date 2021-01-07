//minimal command line metal compute shader
//  Updated from Matthew Hamilton's https://gist.github.com/mhamilt/a5c2bbb02684e5db362712c9be7a02ca
//
// xcrun -sdk macosx metal -c add.metal -o add.air
// xcrun -sdk macosx metallib add.air -o default.metallib
// clang -framework Foundation -framework metal main.m -o hello  -Wl,-sectcreate,addseg,addsect,default.metallib
// ./hello
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
	@autoreleasepool {   
        //----------------------------------------------------------------------
        // Setup
		NSArray *devices = MTLCopyAllDevices();
		id<MTLDevice> device = devices[0];
		if (device == nil) {
			printf("No metal device found\n");
			return 1;
		}
		printf("Running compute application on device %s\n", [device.name UTF8String]);
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLLibrary> library = [device newDefaultLibrary];
		id<MTLFunction> kernelFunction = [library newFunctionWithName:@"add"];
		if (kernelFunction == nil) {
			printf("kernel not loaded\n");
			return 2;
		}
		//----------------------------------------------------------------------
        // pipeline
	    NSError *error = NULL;
        [commandQueue commandBuffer];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:[device newComputePipelineStateWithFunction:kernelFunction error:&error]];
        //----------------------------------------------------------------------
        // Set Data
        float input[] = {11,22};
        printf("input = %g %g\n", input[0], input[1]);
		NSInteger dataSize = sizeof(input);
        [encoder setBuffer:[device newBufferWithBytes:input length:dataSize options:0]
                    offset:0
                   atIndex:0];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:sizeof(float) options:0];
		[encoder setBuffer:outputBuffer offset:0 atIndex:1];
        //----------------------------------------------------------------------
        // Run Kernel
	    MTLSize numThreadgroups = {1,1,1};
        MTLSize numgroups = {1,1,1};
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:numgroups];
	    [encoder endEncoding];
        [commandBuffer commit];
		[commandBuffer waitUntilCompleted];
        //----------------------------------------------------------------------
        // Results
        float *output = [outputBuffer contents];
        printf("result = %f\n", output[0]);
    }
    return 0;
}