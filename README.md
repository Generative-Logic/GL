## Licensing 📜

#test
Generative Logic is dual-licensed under the AGPLv3 and a Commercial License.

**1. AGPLv3 License (Open-Source)**

For open-source projects, academic research, and personal use, Generative Logic is licensed under the AGPLv3. 
This license requires that any derivative works or applications that use this software must also be made open-source under the same terms. The full license text is available in the [LICENSE](LICENSE) file.

**2. Commercial License**

If you wish to use Generative Logic in a proprietary, closed-source commercial application, a commercial license is required. This license frees you from the obligations of the AGPLv3 and includes enterprise-grade features such as a limited warranty and IP indemnity.

For more details and to request a quote, please visit our commercial licensing page:
**https://generative-logic.com/commercial**

**Contributing**

Please note that all contributions to this project require a signed Contributor License Agreement (CLA). For more information, please see our `CONTRIBUTING.md` file.


Run Modes

🔹 Quick Mode (default)

Settings:

parameters.quick_run = True
parameters.min_num_operators_key = 2
parameters.max_iteration_number_proof = 17

The system does not generate theorem candidates.
It receives a predefined list of theorems and attempts to prove all of them.
These inputs are sufficient to prove the six foundational theorems of Peano arithmetic:
Commutativity (addition, multiplication)
Associativity (addition, multiplication)
Left and right distributivity
Output: HTML proof graph in files/full_proof_graph/
Estimated runtime: ~1 minute



🔸 Almost-Full Mode

Settings:

parameters.quick_run = False
parameters.min_num_operators_key = 4
parameters.max_iteration_number_proof = 12

The system generates candidate theorems, then attempts to prove them.
Proves five of the six main Peano theorems (missing commutativity of multiplication):
Commutativity (addition)
Associativity (addition, multiplication)
Left and right distributivity

Output: HTML proof graph in files/full_proof_graph/
Estimated runtime: ~1.5 hours



🔸 Full Mode

Settings:

parameters.quick_run = False
parameters.min_num_operators_key = 2
parameters.max_iteration_number_proof = 15

The system generates all candidate theorems and attempts full closure.
Successfully proves all six of the Peano arithmetic laws:
Commutativity (addition, multiplication)
Associativity (addition, multiplication)
Left and right distributivity

Output: HTML proof graph in files/full_proof_graph/
Estimated runtime: ~1 day