//swiftc -framework Foundation -framework metal mmul.swift -o mmul; ./mmul
import Foundation
import MetalPerformanceShaders
import Accelerate

// Prepare some data
/*
    A[m][p]         A[(m*P+p)*IA]
    B[p][n]         B[(p*N+n)*IB]
    C[m][n] 	    C[(m*N+n)*IC]
	C = A * B
*/
let  m = 65528;//64000;//72000; //<- voxels
let  n = 16; //statistical contrast, e.g "1 0 0" 
let  p = 120; //<- shared: participants
let reps = 1; //times to repeat

let rowsA = m
let columnsA = p
let rowsB = p
let columnsB = n
let rowsC = m
let columnsC = n

let a = UnsafeMutablePointer<Float>.allocate(capacity: rowsA * columnsA)
let arrayA = UnsafeMutableBufferPointer(start: a, count: rowsA * columnsA)
arrayA.assign(repeating: Float(2.0))
for i in 0..<arrayA.count {
	arrayA[i] = Float(drand48())
}
let b = UnsafeMutablePointer<Float>.allocate(capacity: rowsB * columnsB)
let arrayB = UnsafeMutableBufferPointer(start: b, count: rowsB * columnsB)
arrayB.assign(repeating: Float(1.0))
for i in 0..<arrayB.count {
	arrayB[i] = Float(drand48())
}

let c = UnsafeMutablePointer<Float>.allocate(capacity: rowsC * columnsC)
let arrayC = UnsafeMutableBufferPointer(start: c, count: rowsC * columnsC)


//run BLAS
let cBLAS = UnsafeMutablePointer<Float>.allocate(capacity: rowsC * columnsC)
let arrayCBLAS = UnsafeMutableBufferPointer(start: cBLAS, count: rowsC * columnsC)
arrayCBLAS.assign(repeating: Float(1.0))

var mn: Float = 2147483647.0
var tot: Float = 0.0
for _ in 0..<reps {
	let startTime = CFAbsoluteTimeGetCurrent()
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(p), 1.0, a, Int32(p), b, Int32(n), 0.0, cBLAS, Int32(n))
	let elapsed = Float(CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
	mn = min(mn, elapsed);
	tot += elapsed
}

print("BLAS Elapsed time min: ", mn ," mean: ", tot / Float(reps))


// Get the device
//let device = MTLCreateSystemDefaultDevice()! // <- iOS, use MTLCopyAllDevices for macOS
let devices = MTLCopyAllDevices() // get all available devices
let device = devices[0] // randomly selected GPU to use
let commandQueue = device.makeCommandQueue()!
//let library = try device.makeLibrary(filepath: "default.metallib")

//let blitEncoder = commandBuffer.makeBlitCommandEncoder()!

// 1. Prepare managed buffers
//  Metal Matrix data is assumed to be stored in row-major order
let rowBytesA = columnsA * MemoryLayout<Float>.stride
let rowBytesB = columnsB * MemoryLayout<Float>.stride
let rowBytesC = columnsC * MemoryLayout<Float>.stride

//https://developer.apple.com/forums/thread/105534
// suggests storageModePrivate not storageModeManaged, though this causes issues
// matrix columns/rows must be evenly divisible by 8
let bufferA = device.makeBuffer(bytes: arrayA.baseAddress!, length: rowsA * rowBytesA, options: [.storageModeManaged])! 
let bufferB = device.makeBuffer(bytes: arrayB.baseAddress!, length: rowsB * rowBytesB, options: [.storageModeManaged])!
let bufferC = device.makeBuffer(length: rowsC * rowBytesC, options: [.storageModeManaged])!

// 2. Encode matrix multiplication
let descrA = MPSMatrixDescriptor(rows: rowsA, columns: columnsA, rowBytes: rowBytesA, dataType: .float32)
let descrB = MPSMatrixDescriptor(rows: rowsB, columns: columnsB, rowBytes: rowBytesB, dataType: .float32)
let descrC = MPSMatrixDescriptor(rows: rowsC, columns: columnsC, rowBytes: rowBytesC, dataType: .float32)

let matrixA = MPSMatrix(buffer: bufferA, descriptor: descrA)
let matrixB = MPSMatrix(buffer: bufferB, descriptor: descrB)
let matrixC = MPSMatrix(buffer: bufferC, descriptor: descrC)
let matMul = MPSMatrixMultiplication(device: device, resultRows: rowsC, resultColumns: columnsC, interiorColumns: columnsA)

mn = 2147483647.0
tot = 0.0
for _ in 0..<reps {
	let startTime = CFAbsoluteTimeGetCurrent()

	let commandBuffer = commandQueue.makeCommandBuffer()!
	matMul.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
	commandBuffer.commit()
	commandBuffer.waitUntilCompleted()

	let elapsed = Float(CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
	mn = min(mn, elapsed);
	tot += elapsed
}

print("METAL Elapsed time min: ", mn ," mean: ", tot / Float(reps))

// Read results
let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: rowsC * columnsC)
let result = UnsafeBufferPointer(start: resultPointer, count: rowsC * columnsC)
print("A values: [\(arrayA[0])...\(arrayA[columnsA * rowsA - 1])]")
print("B values: [\(b[0])...\(b[columnsB * rowsB - 1])]")
print("BLAS results: [\(cBLAS[0])...\(cBLAS[columnsC * rowsC - 1])]")
print("Metal results: [\(result[0])...\(result[result.count - 1])]")
var largestError: Float = 0.0
for i in 0..<result.count {
	largestError = max(largestError, abs(cBLAS[i] - result[i]))
	if (abs(cBLAS[i] - result[i]) > 30.0) {
	 print(cBLAS[i], " != ", result[i])
	}
}
print("Largest error: \(largestError)")	