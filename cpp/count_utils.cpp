#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <unordered_set>

namespace py = pybind11;

using NGram = std::vector<int>;

struct NGramHash
{
  std::size_t operator()(const NGram &ng) const
  {
    std::size_t seed = ng.size();
    for (auto &i : ng)
    {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct BatchResult
{
  int total_overlap_count;
  int total_samples_with_overlap;
  std::unordered_map<NGram, int, NGramHash> overlapping_ngrams;
};

BatchResult batch_n_gram_overlap(const std::vector<std::vector<int>> &batch_sample_ids,
                                 const std::vector<int> &twin_ids,
                                 int n)
{
  BatchResult result{0, 0, {}};
  std::unordered_map<NGram, int, NGramHash> twin_ngram_counts;

  // Count n-grams in twin
  for (size_t i = 0; i <= twin_ids.size() - n; ++i)
  {
    NGram ngram(twin_ids.begin() + i, twin_ids.begin() + i + n);
    twin_ngram_counts[ngram]++;
  }

  for (const auto &sample_ids : batch_sample_ids)
  {
    std::unordered_map<NGram, int, NGramHash> sample_ngram_counts;
    int sample_overlap = 0;

    // Count n-grams in sample and check for overlap
    for (size_t i = 0; i <= sample_ids.size() - n; ++i)
    {
      NGram ngram(sample_ids.begin() + i, sample_ids.begin() + i + n);
      sample_ngram_counts[ngram]++;

      if (twin_ngram_counts.find(ngram) != twin_ngram_counts.end())
      {
        int overlap = std::min(sample_ngram_counts[ngram], twin_ngram_counts[ngram]);
        sample_overlap += overlap;
        result.overlapping_ngrams[ngram] += overlap;
      }
    }

    if (sample_overlap > 0)
    {
      result.total_overlap_count += sample_overlap;
      result.total_samples_with_overlap++;
    }
  }

  return result;
}

std::unordered_map<int, int64_t> count_tokens(const std::vector<std::vector<int>> &batch_tokens)
{
  std::unordered_map<int, int64_t> token_counts;
  for (const auto &tokens : batch_tokens)
  {
    for (int token : tokens)
    {
      token_counts[token]++;
    }
  }
  return token_counts;
}

struct TokenStats
{
  int64_t tf;
  int64_t df;
};

std::unordered_map<int, TokenStats> count_tf_df(const std::vector<std::vector<int>> &batch_tokens)
{
  std::unordered_map<int, TokenStats> token_stats;
  for (const auto &tokens : batch_tokens)
  {
    std::unordered_set<int> unique_tokens;
    for (int token : tokens)
    {
      token_stats[token].tf++;
      unique_tokens.insert(token);
    }
    for (int token : unique_tokens)
    {
      token_stats[token].df++;
    }
  }
  return token_stats;
}

struct SearchResult
{
  int total_matches;
  std::vector<int> matching_indices;
};

SearchResult batch_search_sequence(const std::vector<std::vector<int64_t>> &batch_input_ids,
                                   const std::vector<int> &search_sequence,
                                   int batch_offset)
{
  SearchResult result{0, {}};

  for (size_t i = 0; i < batch_input_ids.size(); ++i)
  {
    const auto &sample = batch_input_ids[i];

    auto it = std::search(sample.begin(), sample.end(),
                          search_sequence.begin(), search_sequence.end());

    if (it != sample.end())
    {
      result.total_matches++;
      result.matching_indices.push_back(i + batch_offset);
    }
  }

  return result;
}

PYBIND11_MODULE(count_utils, m)
{
  py::class_<BatchResult>(m, "BatchResult")
      .def_readwrite("total_overlap_count", &BatchResult::total_overlap_count)
      .def_readwrite("total_samples_with_overlap", &BatchResult::total_samples_with_overlap)
      .def_property_readonly("overlapping_ngrams", [](const BatchResult &br)
                             {
            py::dict result;
            for (const auto& pair : br.overlapping_ngrams) {
                py::tuple key = py::cast(pair.first);
                result[key] = pair.second;
            }
            return result; });

  m.def("batch_n_gram_overlap", &batch_n_gram_overlap, "A function that computes n-gram overlap for a batch of samples");

  py::class_<TokenStats>(m, "TokenStats")
      .def(py::init<>())
      .def_readwrite("tf", &TokenStats::tf)
      .def_readwrite("df", &TokenStats::df);
  m.def("count_tf_df", &count_tf_df, "A function that counts term frequency and document frequency");

  m.def("count_tokens", &count_tokens, "A function that counts tokens");

  py::class_<SearchResult>(m, "SearchResult")
      .def_readwrite("total_matches", &SearchResult::total_matches)
      .def_readwrite("matching_indices", &SearchResult::matching_indices);
  m.def("batch_search_sequence", &batch_search_sequence,
        "A function that searches for a sequence in a batch of input IDs");
}
