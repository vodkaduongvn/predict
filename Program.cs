using BERTTokenizers;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using Newtonsoft.Json;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";
            var sentence = GetContentPDF(@"C:\WORK\M3\bert-csharp\Z000000002.pdf");
            //var sentence = "Bob Dylan is from Duluth";
            Console.WriteLine(sentence);

            // Create Tokenizer and tokenize the sentence.
            var tokenizer = new BertUncasedLargeTokenizer();
            //var tokenizer = new BertBaseTokenizer();

            // Get the sentence tokens.
            try
            {
                var tokens = tokenizer.Tokenize(sentence);
                // Console.WriteLine(String.Join(", ", tokens));

                // Encode the sentence and pass in the count of the tokens in the sentence.
                var encoded = tokenizer.Encode(tokens.Count(), sentence);

                // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
                var bertInput = new BertInput()
                {
                    InputIds = encoded.Select(t => t.Item1).ToArray(),
                    AttentionMask = encoded.Select(t => t.Item2).ToArray(),
                    TypeIds = encoded.Select(t => t.Item3).ToArray(),
                };

                // Get path to model to create inference session.
                var modelPath = @"C:\WORK\M3\bert-csharp\model.onnx";

                // Create input tensor.
                var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
                var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
                var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);


                // Create input data for session.
                var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", input_ids),
                                         NamedOnnxValue.CreateFromTensor("attention_mask", attention_mask),
                                         NamedOnnxValue.CreateFromTensor("token_type_ids", token_type_ids) };

                // Create an InferenceSession from the Model Path.
                var session = new InferenceSession(modelPath);

                // Run session and send the input data in to get inference output. 
                var output = session.Run(input);

                // Call ToList on the output.
                // Get the First and Last item in the list.
                // Get the Value of the item and cast as IEnumerable<float> to get a list result.
                List<float> startLogits = (output.ToList().First().Value as IEnumerable<float>).ToList();
                List<float> endLogits = (output.ToList().Last().Value as IEnumerable<float>).ToList();

                // Get the Index of the Max value from the output lists.
                var startIndex = startLogits.ToList().IndexOf(startLogits.Max());
                var endIndex = endLogits.ToList().IndexOf(endLogits.Max());

                // From the list of the original tokens in the sentence
                // Get the tokens between the startIndex and endIndex and convert to the vocabulary from the ID of the token.
                var predictedTokens = tokens
                            .Skip(startIndex)
                            .Take(endIndex + 1 - startIndex)
                            .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                            .ToList();

                // Print the result.
                Console.WriteLine(String.Join(" ", predictedTokens));
            }
            catch (Exception ex)
            {

            }
        }

        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            // Create a tensor with the shape the model is expecting. Here we are sending in 1 batch with the inputDimension as the amount of tokens.
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                // Add each to the input Tenor result.
                // Set index and array value of each input Tensor.
                input[0, i] = inputArray[i];
            }
            return input;
        }

        private static string GetContentPDF(string pdfFilePath)
        {
            //string pdfFilePath = "path_to_your_pdf_file.pdf";
            string text = "";
            try
            {
                using (PdfDocument pdfDocument = new PdfDocument(new PdfReader(pdfFilePath)))
                {
                    for (int pageNum = 1; pageNum <= pdfDocument.GetNumberOfPages(); pageNum++)
                    {
                        PdfPage page = pdfDocument.GetPage(pageNum);
                        ITextExtractionStrategy strategy = new SimpleTextExtractionStrategy();
                        text = PdfTextExtractor.GetTextFromPage(page, strategy);
                        var formattedText = text.Replace("\n", "");
                        text = formattedText;
                        Console.WriteLine($"Page {pageNum}:\n{text}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }

            return text;
        }
    }

    public class BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }


    
}