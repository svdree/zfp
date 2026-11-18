#include <assert.h>

#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "ErrorCheck.h"

#include "pointers.cuh"
#include "type_info.cuh"
#include <iostream>
#include <vector>
#include <assert.h>

// we need to know about bitstream, but we don't 
// want duplicate symbols.
#ifndef inline_
  #define inline_ inline
#endif

#include "zfp/bitstream.inl"
namespace internal 
{ 
  
bool is_contigous3d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[3];
  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1) + 
               std::min(stride.z,0) * (idims[2] - 1);

  int64 imax = std::max(stride.x,0) * (idims[0] - 1) + 
               std::max(stride.y,0) * (idims[1] - 1) + 
               std::max(stride.z,0) * (idims[2] - 1);
  offset = imin;
  int64 ns = idims[0] * idims[1] * idims[2];

  return (imax - imin + 1 == ns);
}

bool is_contigous2d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[2];
  idims[0] = dims[0];
  idims[1] = dims[1];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1);

  int64  imax = std::max(stride.x,0) * (idims[0] - 1) + 
                std::max(stride.y,0) * (idims[1] - 1); 

  offset = imin;
  return (imax - imin + 1) == (idims[0] * idims[1]);
}

bool is_contigous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if(stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contigous(const uint dims[3], const int3 &stride, long long int &offset)
{
  int d = 0;
  
  if(dims[0] != 0) d++;
  if(dims[1] != 0) d++;
  if(dims[2] != 0) d++;

  if(d == 3)
  {
    return is_contigous3d(dims, stride, offset);
  }
  else if(d == 2)
  {
   return is_contigous2d(dims, stride, offset);
  }
  else
  {
    return is_contigous1d(dims[0], stride.x, offset);
  } 

}
//
// encode expects device pointers
//
template<typename T>
size_t encode(uint dims[3], int3 stride, int bits_per_block, T *d_data, Word *d_stream)
{

  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;
  if(d == 1)
  {
    int dim = dims[0];
    int sx = stride.x;
    stream_size = cuZFP::encode1<T>(dim, sx, d_data, d_stream, bits_per_block); 
  }
  else if(d == 2)
  {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    stream_size = cuZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if(d == 3)
  {
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    stream_size = cuZFP::encode<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");
  
  return stream_size; 
}

template<typename T>
size_t decode(uint ndims[3], int3 stride, int bits_per_block, Word *stream, T *out)
{

  int d = 0;
  size_t out_size = 1;
  size_t stream_bytes = 0;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  if(d == 3)
  {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);

    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 

    stream_bytes = cuZFP::decode3<T>(dims, s, stream, out, bits_per_block); 
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    int sx = stride.x;

    stream_bytes = cuZFP::decode1<T>(dim, sx, stream, out, bits_per_block); 

  }
  else if(d == 2)
  {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 

    stream_bytes = cuZFP::decode2<T>(dims, s, stream, out, bits_per_block); 
  }
  else std::cerr<<" d ==  "<<d<<" not implemented\n";
 
  return stream_bytes;
}

Word *setup_device_stream_compress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(bitstream_word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
  cudaMalloc(&d_stream, max_size);
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(bitstream_word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  //TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
  cudaMalloc(&d_stream, size);
  cudaMemcpy(d_stream, stream->stream->begin, size, cudaMemcpyHostToDevice);
  return d_stream;
}

void * offset_void(zfp_type type, void *ptr, long long int offset)
{
  void * offset_ptr = NULL;
  if(type == zfp_type_float)
  {
    float* data = (float*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_double)
  {
    double* data = (double*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int32)
  {
    int * data = (int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int64)
  {
    long long int * data = (long long int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  return offset_ptr;
}

void *setup_device_field_compress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
    return field->data;
  }
  
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contigous(dims, stride, offset);
  
  void * host_ptr = offset_void(field->type, field->data, offset);;

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
    cudaMalloc(&d_data, field_bytes);

    cudaMemcpy(d_data, host_ptr, field_bytes, cudaMemcpyHostToDevice);
  }
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
    return field->data;
  }

  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contigous(dims, stride, offset);

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
    cudaMalloc(&d_data, field_bytes);
  }
  return offset_void(field->type, d_data, -offset);
}

void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  bool device = cuZFP::is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if(bytes > 0)
  {
    cudaMemcpy(h_offset_ptr, d_offset_ptr, bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_offset_ptr);
}

//
// Fixed-precision (variable-rate) decode support.
//
// Unlike fixed-rate mode, a fixed-precision block's compressed length isn't
// known ahead of time -- it depends on the block's own common exponent
// combined with the stream's maxprec/minexp (see precision() in
// decode.cuh). So before the parallel GPU decode kernel can look up each
// block's starting bit offset, a sequential pre-pass walks the whole
// (host-resident) stream once, measuring how many bits each block actually
// consumed, without reconstructing any values. That pass is inherently
// sequential -- there's no way to know block N's offset without having
// walked blocks 0..N-1 first -- but it's much cheaper than a full decode
// since it never materializes coefficients, just counts bits.

// Minimal sequential bit reader over a host-resident zfp bitstream buffer.
// Mirrors the bit layout cuZFP::BlockReader (decode.cuh) reads on the
// device, but walks forward through the whole stream instead of jumping to
// a precomputed offset.
class HostBitReader
{
public:
  explicit HostBitReader(const Word *buffer)
    : m_words(buffer), m_buffer(*buffer), m_bit(0), m_bits_read(0)
  {}

  uint read_bit()
  {
    uint bit = (uint)(m_buffer & 1);
    advance(1);
    return bit;
  }

  // n must be <= bits per Word
  uint64 read_bits(uint n)
  {
    uint64 value = 0;
    uint got = 0;
    while (got < n) {
      uint take = (uint)(sizeof(Word) * 8) - m_bit;
      if (take > n - got)
        take = n - got;
      Word mask = (take >= sizeof(Word) * 8) ? ~(Word)0 : (((Word)1 << take) - 1);
      value |= ((uint64)(m_buffer & mask)) << got;
      m_buffer >>= take;
      m_bit += take;
      got += take;
      if (m_bit >= sizeof(Word) * 8) {
        m_bit = 0;
        ++m_words;
        m_buffer = *m_words;
      }
    }
    m_bits_read += n;
    return value;
  }

  size_t bits_read() const { return m_bits_read; }

private:
  void advance(uint n)
  {
    m_buffer >>= n;
    m_bit += n;
    m_bits_read += n;
    if (m_bit >= sizeof(Word) * 8) {
      m_bit = 0;
      ++m_words;
      m_buffer = *m_words;
    }
  }

  const Word *m_words;
  Word m_buffer;
  uint m_bit;
  size_t m_bits_read;
};

// Consume (without materializing) exactly the bits that
// cuZFP::decode_ints_prec (decode.cuh) / src/template/decode.c's
// decode_few_ints_prec would consume for one block's integer coefficients.
// The control flow depends only on the bits read, not on their decoded
// values, so this is a safe, cheaper stand-in for measurement purposes.
void measure_ints_prec(HostBitReader &reader, int intprec, int maxprec, uint size)
{
  int kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint n = 0;
  for (int k = intprec; k-- > kmin;) {
    reader.read_bits(n);
    for (; n < size && reader.read_bit(); n++) {
      for (; n < size - 1 && !reader.read_bit(); n++)
        ;
    }
  }
}

// Measure the bits consumed by one block, mirroring
// src/template/decodef.c's decode_block (floats) / decodei.c's
// zfp_decode_block (integers), and cuZFP::zfp_decode_precision's device-side
// counterpart. Returns the number of bits consumed.
template<typename Scalar>
size_t measure_block_bits(HostBitReader &reader, uint minbits, int maxprec, int minexp, int dims)
{
  size_t start = reader.bits_read();
  uint bits = cuZFP::is_int<Scalar>() ? 0u : 1u;
  bool nonzero = true;

  if (!cuZFP::is_int<Scalar>())
    nonzero = reader.read_bit() != 0;

  if (nonzero) {
    int mp = maxprec;
    if (!cuZFP::is_int<Scalar>()) {
      uint ebits = (uint)cuZFP::get_ebits<Scalar>();
      int emax = (int)reader.read_bits(ebits) - cuZFP::get_ebias<Scalar>();
      bits += ebits;
      mp = cuZFP::precision(emax, maxprec, minexp, dims);
    }
    size_t before_ints = reader.bits_read();
    measure_ints_prec(reader, cuZFP::get_precision<Scalar>(), mp, 1u << (2 * dims));
    uint ints_bits = (uint)(reader.bits_read() - before_ints);
    uint remaining_minbits = minbits > bits ? minbits - bits : 0;
    if (ints_bits < remaining_minbits)
      reader.read_bits(remaining_minbits - ints_bits);
  }
  else {
    if (minbits > bits)
      reader.read_bits(minbits - bits);
  }

  return reader.bits_read() - start;
}

// Phase 1: sequential host-side scan building an exclusive-prefix-sum table
// of per-block starting bit offsets (size num_blocks + 1, with the last
// entry being the total bits consumed by the whole field).
template<typename Scalar>
std::vector<uint64> scan_block_offsets_precision(const Word *h_stream, size_t num_blocks, uint minbits, int maxprec, int minexp, int dims)
{
  std::vector<uint64> offsets(num_blocks + 1);
  HostBitReader reader(h_stream);
  for (size_t i = 0; i < num_blocks; ++i) {
    offsets[i] = (uint64)reader.bits_read();
    measure_block_bits<Scalar>(reader, minbits, maxprec, minexp, dims);
  }
  offsets[num_blocks] = (uint64)reader.bits_read();
  return offsets;
}

// Phase 2: dispatch the parallel per-block GPU decode by dimensionality,
// given the offset table computed above.
template<typename T>
void decode_precision(uint dims[3], int3 stride, int maxprec, int minexp,
                       const uint64 *d_block_offsets, size_t num_blocks,
                       Word *d_stream, T *d_data)
{
  int d = 0;
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0) d++;

  if (d == 1) {
    uint dim = dims[0];
    cuZFP::decode1_precision<T>(dim, stride.x, d_stream, d_block_offsets, num_blocks, d_data, maxprec, minexp);
  }
  else if (d == 2) {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x;
    s.y = stride.y;
    cuZFP::decode2_precision<T>(ndims, s, d_stream, d_block_offsets, num_blocks, d_data, maxprec, minexp);
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    cuZFP::decode3_precision<T>(ndims, stride, d_stream, d_block_offsets, num_blocks, d_data, maxprec, minexp);
  }
  else {
    std::cerr << "cuda fixed-precision decode: unsupported dimensionality (" << d << ")\n";
  }
}

// Number of 4^d blocks (padded up to a multiple of 4 per dimension), matching
// the padding math in decode{1,2,3}launch.
size_t num_blocks_precision(const uint dims[3], int d)
{
  uint padded[3] = {dims[0], dims[1], dims[2]};
  for (int i = 0; i < d; ++i)
    if (padded[i] % 4 != 0)
      padded[i] += 4 - padded[i] % 4;

  size_t blocks = 1;
  for (int i = 0; i < d; ++i)
    blocks *= padded[i] / 4;
  return blocks;
}

// Runs phase 1 (host scan) then phase 2 (parallel device decode) for one
// field, and returns the number of stream bytes consumed -- analogous to
// the return value of internal::decode<T> for the fixed-rate path -- so the
// caller can position the zfp_stream the same way afterwards.
template<typename Scalar>
size_t decompress_precision(const Word *h_stream, Word *d_stream, uint dims[3], int d,
                             int3 stride, size_t num_blocks, uint minbits, int maxprec,
                             int minexp, Scalar *d_data)
{
  std::vector<uint64> offsets =
    scan_block_offsets_precision<Scalar>(h_stream, num_blocks, minbits, maxprec, minexp, d);

  uint64 *d_offsets = NULL;
  cudaMalloc(&d_offsets, offsets.size() * sizeof(uint64));
  cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint64), cudaMemcpyHostToDevice);

  decode_precision<Scalar>(dims, stride, maxprec, minexp, d_offsets, num_blocks, d_stream, d_data);

  cudaFree(d_offsets);

  size_t total_bits = (size_t)offsets.back();
  return (total_bits + Wsize - 1) / Wsize * sizeof(Word);
}

} // namespace internal

size_t
cuda_compress(zfp_stream *stream, const zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;
  
  size_t stream_bytes = 0;
  long long int offset = 0; 
  void *d_data = internal::setup_device_field_compress(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    stream_bytes = internal::encode<float>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    stream_bytes = internal::encode<double>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    stream_bytes = internal::encode<int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) d_data;
    stream_bytes = internal::encode<long long int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }

  internal::cleanup_device_ptr(stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, 0, offset, field->type);

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  size_t compressed_size = stream_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + compressed_size;

  return stream_bytes;
}
  
void 
cuda_decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
   
  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  long long int offset = 0;
  void *d_data = internal::setup_device_field_decompress(field, stride, offset);
  
  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

   
  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }
  
  size_t bytes = type_size * field_size;
  internal::cleanup_device_ptr(stream->stream->begin, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes, offset, field->type);
  
  // this is how zfp determines if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + words_read;
}

void
cuda_decompress_precision(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int d = 0;
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0) d++;

  // Blocks are decoded down to maxprec bit planes with no bit-budget check
  // (see decode_ints_prec in decode.cuh); this mirrors the "_prec" path the
  // serial decoder takes whenever maxbits doesn't actually constrain a
  // block (with_maxbits() in src/template/codec.c). Streams configured via
  // the public zfp_stream_set_precision() always have maxbits fixed at the
  // ZFP_MAX_BITS sentinel, which never binds for these block sizes, so this
  // is just a safety net against unusual hand-configured streams.
  uint block_size = 1u << (2 * d);
  if ((stream->maxprec + 1) * block_size - 1 > stream->maxbits)
  {
    std::cerr << "cuda fixed-precision decode: maxbits unexpectedly constrains this stream; unsupported\n";
    return;
  }

  int3 stride;
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  long long int offset = 0;
  void *d_data = internal::setup_device_field_decompress(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return;
  }

  // Phase 1 (below) needs the compressed bytes on the host; stage a copy if
  // the stream is already device-resident. This is a one-time cost, unlike
  // the per-block scan itself.
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  size_t max_stream_bytes = zfp_stream_maximum_size(stream, field);
  std::vector<Word> host_stream_storage;
  const Word *h_stream;
  if(stream_device)
  {
    host_stream_storage.resize((max_stream_bytes + sizeof(Word) - 1) / sizeof(Word));
    cudaMemcpy(host_stream_storage.data(), stream->stream->begin, max_stream_bytes, cudaMemcpyDeviceToHost);
    h_stream = host_stream_storage.data();
  }
  else
  {
    h_stream = (const Word*)stream->stream->begin;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);

  size_t num_blocks = internal::num_blocks_precision(dims, d);
  size_t decoded_bytes = 0;

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    decoded_bytes = internal::decompress_precision<float>(h_stream, d_stream, dims, d, stride, num_blocks, stream->minbits, (int)stream->maxprec, stream->minexp, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    decoded_bytes = internal::decompress_precision<double>(h_stream, d_stream, dims, d, stride, num_blocks, stream->minbits, (int)stream->maxprec, stream->minexp, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    decoded_bytes = internal::decompress_precision<int>(h_stream, d_stream, dims, d, stride, num_blocks, stream->minbits, (int)stream->maxprec, stream->minexp, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    decoded_bytes = internal::decompress_precision<long long int>(h_stream, d_stream, dims, d, stride, num_blocks, stream->minbits, (int)stream->maxprec, stream->minexp, data);
    d_data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  size_t bytes = type_size * field_size;
  internal::cleanup_device_ptr(stream->stream->begin, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes, offset, field->type);

  // this is how zfp determines if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + words_read;
}
