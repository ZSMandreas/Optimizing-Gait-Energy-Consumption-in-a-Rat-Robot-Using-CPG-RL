################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Startup/subdir.mk
-include Src/subdir.mk
-include Drivers/STM32F7xx_HAL_Driver/Src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
Spine\ R4.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
Spine\ R4.list \

OBJCOPY_HEX += \
Spine\ R4.hex \

OBJCOPY_BIN += \
Spine\ R4.bin \


# All Target
all:
	+@make --no-print-directory main-build && make --no-print-directory post-build

# Main-build Target
main-build: Spine\ R4.elf secondary-outputs

# Tool invocations
Spine\ R4.elf: $(OBJS) $(USER_OBJS) C:\Users\schul\Documents\Uni\SS20\Masterarbeit\HBPMouse\HBPMouse\Spine\ R4\STM32F722RETX_FLASH.ld
	arm-none-eabi-gcc -o "Spine R4.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m7 -T"C:\Users\schul\Documents\Uni\SS20\Masterarbeit\HBPMouse\HBPMouse\Spine R4\STM32F722RETX_FLASH.ld" --specs=nosys.specs -Wl,-Map="Spine R4.map" -Wl,--gc-sections -static --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

Spine\ R4.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "Spine R4.list"
	@echo 'Finished building: $@'
	@echo ' '

Spine\ R4.hex: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O ihex $(EXECUTABLES) "Spine R4.hex"
	@echo 'Finished building: $@'
	@echo ' '

Spine\ R4.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "Spine R4.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) *
	-@echo ' '

post-build:
	rm ../Debug/Src/uart-ShowVersion.o
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_HEX) $(OBJCOPY_BIN)

.PHONY: all clean dependents
.SECONDARY: post-build

-include ../makefile.targets
