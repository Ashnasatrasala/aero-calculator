import streamlit as st
import sympy as sp
import re
from pint import UnitRegistry
from hybrid_solver import load_formulas

ureg = UnitRegistry()
Q_ = ureg.Quantity

# ---------------------------------------
# COMPACT UI CSS
# ---------------------------------------
st.set_page_config(page_title="Aero Calculator", layout="wide")

# Make input fields small + horizontal-friendly
st.markdown("""
<style>
    .stTextInput > div > div > input {
        height: 1.8rem;
        padding: 2px 6px;
        font-size: 0.9rem;
    }
    /* FIX TITLE CUTTING */
    .block-container {
        padding-top: 2rem;   /* was 0.5rem */
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# No page title (user requested)
# st.title("Aerospace Engineering Calculator")  <-- removed

# ---------------------------------------------------
# TWO MAIN COLUMNS: LEFT (input) | RIGHT (output)
# ---------------------------------------------------
left, right = st.columns([1.2, 1])

# ---------------------------------------------------
# FORMULA SELECTION
# ---------------------------------------------------
with left:
    st.subheader("Select Formula")
    formulas = load_formulas("formulas.txt")
    formula_texts = [f[0] for f in formulas]

    selected_formula = st.selectbox("", ["-- select --"] + formula_texts)

# ---------------------------------------------------
# IF FORMULA IS SELECTED
# ---------------------------------------------------
if selected_formula != "-- select --":

    lhs_text, rhs_text = [p.strip() for p in selected_formula.split("=")]

    var_candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", selected_formula)
    banned = {"log", "sin", "cos", "tan", "exp", "sqrt", "pi"}
    vars_clean = [v for v in var_candidates if v not in banned]
    symbols_dict = {v: sp.Symbol(v) for v in vars_clean}

    # Safe parsing
    try:
        lhs = sp.sympify(lhs_text, locals=symbols_dict)
        rhs = sp.sympify(rhs_text, locals=symbols_dict)
        expr = lhs - rhs
    except:
        st.error("Formula parse error")
        st.stop()

    variables = sorted(list(expr.free_symbols), key=lambda x: str(x))

    # ---------------------------------------------------
    # HORIZONTAL INPUT ROW
    # ---------------------------------------------------
    with left:
        st.write("Enter values (leave one blank):")

        # Horizontal columns (up to 5 across)
        cols = st.columns(len(variables))

        user_inputs = {}
        blank_var = None

        for i, var in enumerate(variables):
            val = cols[i].text_input(f"{var}", placeholder="value or value unit")

            if val.strip() == "":
                blank_var = str(var)
            else:
                try:
                    if re.fullmatch(r"[0-9.]+", val.strip()):
                        qty = Q_(float(val.strip()), "")
                    else:
                        qty = ureg.parse_expression(val)
                    user_inputs[str(var)] = qty
                except:
                    st.error(f"Invalid input for {var}")

        solve_pressed = left.button("Solve", use_container_width=True)

    # ---------------------------------------------------
    # OUTPUT SECTION (Right Side)
    # ---------------------------------------------------
    with right:
        st.subheader("Output")

        if solve_pressed:

            if blank_var is None:
                st.error("Leave one variable blank.")
            else:
                try:
                    # SI conversion
                    subs = {}
                    for v, qty in user_inputs.items():
                        q_si = qty.to_base_units()
                        subs[sp.Symbol(v)] = float(q_si.magnitude)

                    target = sp.Symbol(blank_var)
                    sol = sp.solve(expr, target)

                    if not sol:
                        st.error("Cannot solve.")
                    else:
                        sol_expr = sol[0]
                        result = sol_expr.subs(subs).evalf()

                        st.success(f"{blank_var} = {result}")
                        st.code(f"{blank_var} = {sol_expr}")

                except Exception as e:
                    st.error(f"Error: {e}")
