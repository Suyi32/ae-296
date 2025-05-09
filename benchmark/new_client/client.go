package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type Instruction struct {
	Instruction       string `json:"instruction"`
	Input             string `json:"input"`
	Output            string `json:"output"`
	InstructionTokens int    `json:"instruction_tokens"`
	InputTokens       int    `json:"input_tokens"`
	OutputTokens      int    `json:"output_tokens"`
}

func sendRequest(prompt string, prompt_len int, output_max_len int, ch chan []float64) {
	var first_token_latency float64
	output_tokens := 0
	url := "http://localhost:8080/generate_stream"

	requestBody, _ := json.Marshal(map[string]interface{}{
		"inputs": prompt,
		"parameters": map[string]interface{}{
			"max_new_tokens":    output_max_len,
			"frequency_penalty": 1,
			"do_sample":         false,
			"ignore_eos":        true,
		},
		"model_dir": "huggyllama/llama-7b",
		// "lora_dir": 'dummy-lora-7b-rank-8-{}'.format(entry_id%32),
		"lora_id": 0,
	})

	client := &http.Client{
		Timeout: 3 * time.Hour,
	}

	request_start_time := time.Now()
	for {
		req, _ := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "Benchmark Client")

		resp, err := client.Do(req)
		if err != nil {
			panic(err)
		}

		defer resp.Body.Close()
		flag := false
		buf := make([]byte, 1024) // buffer size
		for {
			n, err := resp.Body.Read(buf)
			if (err != nil && err != io.EOF) || n == 0 {
				flag = true
				break
			}

			if first_token_latency == 0 {
				first_token_latency = time.Since(request_start_time).Seconds()
			}

			str_chunk := string(buf[:n])
			output_tokens++

			if strings.Contains(str_chunk, "\"finished\": true") {
				flag = true
				break
			}
		}

		if flag {
			break
		} else {
			first_token_latency = 0
			break
		}
	}

	request_end_time := time.Now()
	request_latency := request_end_time.Sub(request_start_time)
	ch <- []float64{float64(prompt_len), float64(output_tokens), request_latency.Seconds(), first_token_latency}
}

func print_stats(resp_info [][]float64, replay_start time.Time, replay_end time.Time, generated_tokens int) {
	// Assuming resp_info is a slice of slices with each inner slice having 4 elements

	// Extracting individual fields from resp_info
	var resp_lats []float64
	var promt_tok_length []int
	var resp_tok_length []int
	var first_tok_lat []float64

	for _, item := range resp_info {
		fmt.Println(item)
		promt_tok_length = append(promt_tok_length, int(item[0]))
		resp_tok_length = append(resp_tok_length, int(item[1]))
		resp_lats = append(resp_lats, item[2])
		first_tok_lat = append(first_tok_lat, item[3])
	}

	// Printing the counts
	fmt.Printf("Num of Reqs: %d\n", len(resp_lats))
	fmt.Printf("Num of Resps: %d\n", len(resp_tok_length))

	// Assuming replay_start and replay_end are time.Time objects
	totalBenchmarkingLatency := replay_end.Sub(replay_start).Seconds()
	fmt.Printf("Total Benchmarking Latency: %.3f s\n", totalBenchmarkingLatency)

	// Calculating average latency per token
	var avgPerTokenLatency float64
	for i := 0; i < len(resp_lats); i++ {
		avgPerTokenLatency += resp_lats[i] / float64(promt_tok_length[i]+resp_tok_length[i])
	}
	avgPerTokenLatency /= float64(len(resp_lats))
	fmt.Printf("Average lat per token: %.3f s\n", avgPerTokenLatency)

	// Calculating average latency per output token
	var avgPerOutputTokenLatency float64
	for i := 0; i < len(resp_lats); i++ {
		avgPerOutputTokenLatency += resp_lats[i] / float64(resp_tok_length[i])
	}
	avgPerOutputTokenLatency /= float64(len(resp_lats))
	fmt.Printf("Average lat per output token: %.3f s\n", avgPerOutputTokenLatency)

	// Calculating average latency per request
	var totalRespLats float64
	for _, lat := range resp_lats {
		totalRespLats += lat
	}
	avgLatencyPerReq := totalRespLats / float64(len(resp_lats))
	fmt.Printf("Avg Latency Per Req: %.6f s\n", avgLatencyPerReq)

	// Assuming generated_tokens is the total number of generated tokens
	totalGeneratedTokens := 0
	for _, length := range resp_tok_length {
		totalGeneratedTokens += length
	}
	fmt.Printf("Total number of generated tokens: %d (dataset); %d (resp)\n", generated_tokens, totalGeneratedTokens)

	// Calculating average first token latency
	var totalFirstTokLat float64
	for _, lat := range first_tok_lat {
		totalFirstTokLat += lat
	}
	avgFirstTokenLatency := totalFirstTokLat / float64(len(first_tok_lat))
	fmt.Printf("Average first token latency: %.3f s\n", avgFirstTokenLatency)
}

func main() {
	trace_fn := flag.String("trace_dir", "trace.txt", "the file to the invocation time")
	data_fn := flag.String("data_dir", "alpaca_data.json", "the file to alpaca data")
	max_prompt_len := flag.Int("max_prompt_len", 16, "the max_prompt_len")
	max_output_len := flag.Int("max_output_len", 64, "the max_output_len")
	fmt.Println("max_prompt_len:", max_prompt_len, "max_output_len", max_output_len)
	flag.Parse()

	// read invocation trace
	trace_file, err := os.Open(*trace_fn)
	if err != nil {
		log.Fatal(err)
	}
	defer trace_file.Close()
	var invoke_times []float64
	scanner := bufio.NewScanner(trace_file)
	for scanner.Scan() {
		line := scanner.Text()
		if cur_time, err := strconv.ParseFloat(strings.TrimSpace(line), 64); err == nil {
			invoke_times = append(invoke_times, cur_time)
		}
	}

	// read data
	data, err := os.ReadFile(*data_fn)
	if err != nil {
		log.Fatal(err)
	}
	var alpaca_insts []Instruction
	err = json.Unmarshal(data, &alpaca_insts)
	if err != nil {
		log.Fatal(err)
	}

	// run trace replay
	inst_id := 0
	resultChannel := make(chan []float64, len(invoke_times))
	start_time := time.Now()
	for i := 0; i < len(invoke_times); i++ {
		duration := time.Duration(0)
		if i == 0 {
			duration = time.Duration(invoke_times[i]*1000*1000) * time.Microsecond
		} else {
			duration = time.Duration((invoke_times[i]-invoke_times[i-1])*1000*1000) * time.Microsecond
		}
		time.Sleep(duration)
		prompt := "dummy"
		output_len := 0
		for ; ; inst_id++ {
			prompt = alpaca_insts[inst_id].Instruction + " " + alpaca_insts[inst_id].Input
			if alpaca_insts[inst_id].OutputTokens < *max_output_len &&
				(alpaca_insts[inst_id].InstructionTokens+alpaca_insts[inst_id].InputTokens < *max_prompt_len) {
				output_len = alpaca_insts[inst_id].OutputTokens
				break
			}
		}
		go sendRequest(prompt, len(prompt), output_len, resultChannel)
		fmt.Println("Sent request #:", i, alpaca_insts[inst_id])
		inst_id++
	}

	generated_tokens := 0
	// Collect the results
	var results [][]float64
	for i := 0; i < len(invoke_times); i++ {
		result := <-resultChannel
		results = append(results, result)
		generated_tokens += int(result[1])
	}
	end_time := time.Now()

	print_stats(results, start_time, end_time, generated_tokens)

	// Print the collected results
	fmt.Println(results)
}
