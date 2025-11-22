addresses = [0, 16, 32, 48, 64, 0, 16, 128, 256, 0,  ]  
block_size = 16
cache_size = 2 * 1024  
num_cache_blocks = cache_size // block_size  
blocks = []
for addr in addresses:
    block_number = addr // block_size
    blocks.append(block_number)

print("Блок нөмірлері:", blocks)
print()


direct_cache = [-1] * num_cache_blocks  
hit_direct = 0
miss_direct = 0

for block in blocks:
    index = block % num_cache_blocks
    if direct_cache[index] == block:
        print(f"Блок {block}: HIT")
        hit_direct += 1
    else:
        print(f"Блок {block}: MISS (жүктелді)")
        direct_cache[index] = block
        miss_direct += 1

print("\n=== Direct Mapped Cache Нәтиже ===")
print("Hit саны:", hit_direct)
print("Miss саны:", miss_direct)
print("Hit ratio:", hit_direct / (hit_direct + miss_direct))
print()

fully_cache = []
hit_full = 0
miss_full = 0

for block in blocks:
    if block in fully_cache:
        print(f"Блок {block}: HIT")
        hit_full += 1
    else:
        print(f"Блок {block}: MISS (жүктелді)")
        fully_cache.append(block)
        miss_full += 1

print("\n=== Fully Associative Cache Нәтиже ===")
print("Hit саны:", hit_full)
print("Miss саны:", miss_full)
print("Hit ratio:", hit_full / (hit_full + miss_full))